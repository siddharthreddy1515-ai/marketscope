# evidently_monitor.py
import os
import sys
import argparse
import asyncio
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from workflow import run_workflow
from companies import COMPANIES

DRIFT_SHARE_THRESHOLD = 0.5
CACHE_FILE = "current_reports.csv"

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["report_text"].fillna("")
    features = pd.DataFrame()
    features["char_length"] = text.str.len().astype(float)
    features["word_count"] = text.str.split().str.len().astype(float)
    features["bullet_count"] = text.str.count(r"(?m)^- ").astype(float)
    features["digit_count"] = text.str.count(r"\d").astype(float)
    features["avg_sentence_length"] = text.apply(
        lambda x: len(x.split()) / max(1, x.count(".") + x.count("!") + x.count("?"))
    )
    return features

def pad_report(report: str) -> str:
    return report + "\n\n" + "\n\n".join(
        ["- Simulated drift bullet point to increase text length."] * 5
    )

async def generate_current_reports(refresh: bool = False) -> pd.DataFrame:
    if os.path.exists(CACHE_FILE) and not refresh:
        print(f"✅ Using cached reports ({CACHE_FILE}). Pass --refresh to regenerate.")
        return pd.read_csv(CACHE_FILE)

    os.environ["MOCK_TAVILY"] = "true"
    print("Generating current reports (calls the LLM, costs money)...")
    results = []
    for company in COMPANIES:
        result = await run_workflow(company)
        results.append({"company": company, "report_text": result["report"]})

    df = pd.DataFrame(results)
    df.to_csv(CACHE_FILE, index=False)
    return df

def read_drift_share(result: dict):
    for m in result.get("metrics", []):
        if "DriftedColumnsCount" in m.get("metric_name", ""):
            value = m.get("value", {})
            return value.get("count"), value.get("share")
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-drift", action="store_true")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cache and regenerate reports from the LLM")
    args = parser.parse_args()

    print("Loading baseline data...")
    if not os.path.exists("baseline_reports.csv"):
        print("Error: baseline_reports.csv not found. Run build_baseline.py first.")
        sys.exit(2)

    baseline_df = pd.read_csv("baseline_reports.csv")
    current_df = asyncio.run(generate_current_reports(args.refresh))
    
    if args.simulate_drift:
        current_df = current_df.copy()
        current_df["report_text"] = current_df["report_text"].apply(pad_report)

    print("Extracting numeric features...")
    ref_features = extract_features(baseline_df)
    cur_features = extract_features(current_df)

    print("Running Evidently Drift Report...")
    report = Report([DataDriftPreset()])
    my_eval = report.run(current_data=cur_features, reference_data=ref_features)

    out_name = "drift_report_simulated.html" if args.simulate_drift else "drift_report.html"
    my_eval.save_html(out_name)
    print(f"✅ Saved drift report to {out_name}")

    count, share = read_drift_share(my_eval.dict())
    if share is None:
        print("❌ Could not find DriftedColumnsCount in result.")
        sys.exit(2)

    print(f"Drifted columns: {int(count)} of {len(ref_features.columns)} (share={share:.2f})")

    if share >= DRIFT_SHARE_THRESHOLD:
        print("⚠️ DRIFT DETECTED!")
        sys.exit(1)
    print("✅ No drift detected. Data is stable.")

if __name__ == "__main__":
    main()