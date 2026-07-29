# XDecomposer API

Install the two HTTP dependencies, then start the service from the repository root:

```bash
pip install fastapi uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/` for the simple browser interface: paste or upload XRD CSV/TXT data, inspect candidate `mp-id` values, download JSON/CSV results, and download each matched structure as CIF. `/docs` remains available for programmatic API debugging.

`POST /v1/decompose` accepts an XRD intensity array. `two_theta` is optional; without it, the input is interpreted as uniformly sampled between 10° and 80°.

```bash
curl -X POST 'http://127.0.0.1:8000/v1/decompose?top_k=3' \
  -H 'content-type: application/json' \
  -d '{"intensities": [0.0, 0.2, 1.0, 0.1, 0.0]}'
```

Each returned phase contains its most likely `mp_id`, cosine `match_score`, activity probability, alternatives, and a `structure_url` for its CIF. Add `include_patterns=true` to also receive the 3,500-point decomposed XRD pattern for every phase.

At first startup, the service calculates reference XRD patterns from `MP500.db`; it logs progress every 25 structures and saves the result to `.cache/mp500_reference_bank.pt`. Later starts load this cache directly. The default is 500 reference structures (`XDECOMPOSER_MAX_REFERENCES=500`), which keeps startup time and memory practical. Paths may be overridden with `XDECOMPOSER_CHECKPOINT`, `XDECOMPOSER_MAE_CHECKPOINT`, `XDECOMPOSER_MATERIALS_DB`, `XDECOMPOSER_REFERENCE_CACHE`, and `XDECOMPOSER_MAX_REFERENCES`.
