# MarketScope

MarketScope researches companies and presents the evidence behind its findings. Enter a company name and, if you know it, the official website to help identify the right company.

[Open the hosted dashboard](https://marketscope-4owi.onrender.com)

## What the dashboard shows

- **Core offering** links company and product information to the selected source passage.
- **Recent developments** counts findings with an established event date in the reporting window. Observations with unconfirmed dates are shown alongside the count. Findings classified as historical appear in a separate section.
- **Research coverage** shows how many pages were collected and how many were cited. The source library includes publication dates where available and a **Cited only** filter.
- **Competitive alternatives** shows cited comparisons and labels mentions where product overlap is unconfirmed.

Open the evidence on a card to read the selected passage and visit the source. Use **Download source brief** for an HTML report or **Download evidence JSON** for the selected passages and their references. Citation links let readers check the claim against the source. Coverage depends on what the searches retrieve, and publication dates can differ from event dates.

## Run locally

Docker is sufficient to start the dashboard and health endpoint.

```sh
docker build -t marketscope .
docker run --rm -p 127.0.0.1:8000:8000 marketscope
```

Open the [local dashboard](http://127.0.0.1:8000). Live analysis defaults to disabled. In this mode, the dashboard and health endpoint work without API keys.

For local live research, copy `.env.example` to `.env` and set your own OpenAI and Tavily keys. Stop the first container before starting the live one on the same port.

```sh
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env \
  -e ENABLE_LIVE_ANALYSIS=true -e MOCK_TAVILY=false marketscope
```

Click **Analyze** to run Tavily searches and GPT-4o analysis. Those calls can incur provider charges. Keep `.env` local. Enabling analysis allows any caller who can reach the endpoint to trigger research.

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Dashboard |
| `GET /health` | Service health |
| `POST /api/v1/analyze` | Analyze a company using `company_name` and an optional `official_domain`. Returns HTTP 503 when live analysis is disabled. |

## How it works

1. **Research.** A LangGraph workflow uses Tavily to collect company and competitor sources. When a website is supplied, the workflow uses it to check company relevance.
2. **Analysis.** GPT-4o selects findings and references to passages from those sources. The backend checks the references against the retrieved text.
3. **Presentation.** FastAPI serves the dashboard and structured report. The dashboard uses the selected passages to render findings, KPI evidence, and exports. If the model times out after research completes, the collected sources can still be shown.

The API and dashboard are in [`main.py`](main.py) and [`index.html`](index.html). Research runs in [`workflow.py`](workflow.py), with evidence handling in [`analysis_contract.py`](analysis_contract.py) and routing rules in [`research_policy.py`](research_policy.py).

## Checks and deployment

GitHub Actions runs pytest with saved fixtures and placeholder credentials on pull requests to `main`. CI also builds the Docker image and checks the dashboard, health endpoint, and disabled analysis route with external networking turned off. The live provider evaluation is a separate, manually triggered workflow. [CI workflow](.github/workflows/ci.yml) · [Live smoke workflow](.github/workflows/live-smoke.yml)

The hosted service uses the root [`Dockerfile`](Dockerfile). Verify `/health` and load the dashboard after deployment. Both checks can run without starting an analysis.

## Author

Siddharth · [LinkedIn](https://www.linkedin.com/in/siddharth-donuru-816413290) · [GitHub](https://github.com/siddharthreddy1515-ai)
