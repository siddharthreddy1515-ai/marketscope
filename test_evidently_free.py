# test_evidently_free.py  (costs $0, no LLM calls)
import sys
import json
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def find_count_share(obj):
    """Recursively find a dict holding both 'count' and 'share'."""
    if isinstance(obj, dict):
        if "count" in obj and "share" in obj:
            return obj["count"], obj["share"]
        for v in obj.values():
            c, s = find_count_share(v)
            if s is not None:
                return c, s
    elif isinstance(obj, list):
        for v in obj:
            c, s = find_count_share(v)
            if s is not None:
                return c, s
    return None, None


def read_drift_share(result: dict):
    for m in result.get("metrics", []):
        if "DriftedColumnsCount" in json.dumps(m, default=str):
            return find_count_share(m)
    return None, None


def run_test(name, ref_df, cur_df, expect_drift):
    print(f"\n--- {name} ---")
    my_eval = Report([DataDriftPreset()]).run(current_data=cur_df, reference_data=ref_df)
    my_eval.save_html(f"test_{name}.html")

    result = my_eval.dict()
    print("Top-level keys:", list(result.keys()))
    for m in result.get("metrics", []):
        print("METRIC:", json.dumps(m, default=str)[:300])

    count, share = read_drift_share(result)
    if share is None:
        print("❌ Could not extract drift share. Paste the METRIC lines above.")
        sys.exit(2)

    print(f"Drifted columns: {count} (share={share:.2f})")
    drifted = share >= 0.5
    if drifted != expect_drift:
        print(f"❌ FAIL: expected drift={expect_drift}, got {drifted}")
        sys.exit(1)
    print("✅ PASS")


def main():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({
        "char_length": rng.normal(1000, 50, 40),
        "word_count": rng.normal(160, 10, 40),
        "digit_count": rng.normal(20, 3, 40),
    })
    run_test("stable", ref, ref.copy(), expect_drift=False)

    shifted = ref * 2.5  # clearly different distribution
    run_test("drifted", ref, shifted, expect_drift=True)

    print("\n🎉 Both scenarios behave correctly. Safe to run evidently_monitor.py.")


if __name__ == "__main__":
    main()