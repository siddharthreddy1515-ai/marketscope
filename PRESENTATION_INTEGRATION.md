# Presentation-integrity integration

Dashboard marker: `source-brief-2026-11-19`. API version: `0.3.0`.

Structured V2 reports show selected source passages and citation links. Generated
interpretations, separate executive summaries and raw model reports are excluded
from this display and shared HTML/JSON briefs. The source library remains available
and clearly distinguishes additional retrieved material from selected observations.
Coverage warnings, unknown timing, partial analysis and insufficient evidence remain
visible. A complete analysis status is a parsing outcome, not factual verification.

The original backend lacked the evidence-ID contract required by the presentation
checkpoint. This integration adds the previously reviewed analysis contract and
additive response fields. It uses the existing analysis call, with retries disabled,
1,600 output tokens maximum and a 35-second model timeout. The standalone experimental
writer prompt is not included. Original company/competitor queries, identity
filtering, search limits, research stopping and graph wiring are preserved. Source
purpose annotations and coverage descriptions do not activate new search routing.

Company quotes are linked to exact body spans. The backend validates fingerprints
and source namespaces; the browser rechecks source bodies and Unicode code-point
offsets. Full comparison bodies are not returned, so comparison quote validation
relies on the backend. None of these checks independently proves source truth,
entailment of generated interpretations or completeness of discovered events.

Raw API `report`, finding text and summary fields remain generated drafts for
compatibility. Legacy-format responses retain their earlier display/export path,
with generated-note labels; only `evidence_ids_v2` uses selected-only shared briefs.
Structured V2 JSON downloads use `marketscope-selected-evidence-export` schema
version 1. Consumers of the previous full-response download must adapt; the API
retains existing response fields and adds optional evidence metadata.

The local safe-copy checkpoint package was compared selectively with the original;
it was not copied wholesale. Dependencies, configuration, credentials and local
experiment files are excluded. The original's unrelated uncommitted work is kept
separate from the integration commit.

Offline verification uses mocked providers and blocked external HTTP. It includes
the original backend tests, synthetic real-graph/status tests, eight saved API
responses containing forty selected evidence references, HTML/JSON exports and
stale-report prevention. Saved audits and local verification runners are not
published as application code. Paid judge evaluation and monitoring integration
are separate from these application checks. No new live quality result is claimed.

The most important remaining weakness is selecting enough current, material
developments with passages establishing actor, action, amount, date and status.
Research coverage and material-event selection have not been solved by this patch.
Restart the updated backend and hard-refresh the dashboard to check its build marker.
Publishing the integration branch does not update the hosted demo or merge main.
