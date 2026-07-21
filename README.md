# Financial PDF Extractor

FastAPI backend that takes a company financial-statement PDF (10-K, annual
report, etc.), OCRs it with the **Chandra OCR** model
(`datalab-to/chandra`) via the **Hugging Face Inference API**, then uses
the **Gemini API** to structure the raw text into a standardized JSON
schema, and saves it to disk as `<company>_<timestamp>.json`.

```
PDF upload
   -> render pages to images (PyMuPDF)
   -> Chandra OCR per page (Hugging Face Inference API)
   -> concatenated raw OCR text
   -> Gemini API structures text into target JSON schema
   -> saved as output/<company>_<timestamp>.json
   -> returned in the API response
```

## 1. Setup

```bash
cd financial-pdf-extractor
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the env template and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=your_gemini_api_key
```

- Get an HF token (read access is enough): https://huggingface.co/settings/tokens
- Get a Gemini API key: https://aistudio.google.com/app/apikey

> **Note on Chandra availability:** Chandra is a large vision-language OCR
> model. Whether it's servable through HF's free serverless Inference API
> depends on current HF Inference Providers coverage for that model. If
> `/process-pdf` errors out on the OCR step (e.g. "model not supported by
> any provider"), the fix is either:
> 1. Set `HF_PROVIDER` in `.env` to a provider that hosts it (check the
>    "Deploy" tab on the model page: https://huggingface.co/datalab-to/chandra), or
> 2. Run Chandra locally instead of via the API:
>    `pip install chandra-ocr[hf]` and swap `app/ocr_service.py`'s HF-API
>    call for the local `InferenceManager(method="hf")` call shown in the
>    model card. The rest of the pipeline (Gemini structuring, saving)
>    stays identical.

## 2. Run it

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check it's alive and your keys are picked up:

```bash
curl http://localhost:8000/health
```

## 3. Use it

```bash
curl -X POST http://localhost:8000/process-pdf \
  -F "file=@/path/to/company_annual_report.pdf"
```

Response:

```json
{
  "request_id": "a1b2c3d4",
  "elapsed_seconds": 42.7,
  "saved_path": "output/Nike_Inc_20260719_154213.json",
  "data": { "...": "the extracted JSON" }
}
```

The same JSON is written to `output/<company>_<timestamp>.json`.

Interactive docs: http://localhost:8000/docs

## 4. Output schema

```json
{
  "company_name": "Nike, Inc.",
  "ticker": "NKE",
  "country": "US",
  "currency": "USD",
  "fiscal_year_end": "May 31",
  "source_document": "Nike 10-K (fiscal year ended May 31, 2025)",
  "statement_data": {
    "Current Assets": { "2025-05-31": 23362.0, "2024-05-31": 25382.0 },
    "Total Assets": { "...": "..." },
    "Current Liabilities": { "...": "..." },
    "Total Liabilities": { "...": "..." },
    "Revenue": { "...": "..." },
    "Gross Profit": { "...": "..." },
    "Net Income": { "...": "..." }
  },
  "ratios": {
    "Current Ratio": { "2025": 2.2111, "2024": 2.3961, "source": "..." },
    "Debt Ratio": { "...": "..." },
    "Gross Margin": { "...": "..." },
    "Net Profit Margin": { "...": "..." }
  },
  "notes": "..."
}
```

Field names are normalized by Gemini regardless of what the source PDF
calls them (e.g. "Total Current Assets", "Net Revenue", "Long Term Debt
Obligations" all map to the canonical keys above). See
`app/schema.py` for the full mapping rules and the extra optional fields
(`Total Equity`, `Cash And Equivalents`, `Inventories`, `Long Term Debt`)
that are included when present in the source document.

## 5. Project layout

```
financial-pdf-extractor/
├── app/
│   ├── main.py            FastAPI app & /process-pdf endpoint
│   ├── ocr_service.py      PDF -> images -> Chandra OCR (HF Inference API)
│   ├── gemini_service.py   raw text -> structured JSON (Gemini API)
│   ├── schema.py            canonical schema + prompt instructions
│   ├── utils.py             filename building, JSON saving/parsing
│   └── config.py            env var loading
├── output/                  saved <company>_<timestamp>.json files
├── uploads_tmp/              scratch dir (currently unused, reserved)
├── requirements.txt
├── .env.example
└── README.md
```

## 6. Deploying

No special deployment config is bundled (no Dockerfile) — this is a plain
FastAPI app, so it runs anywhere Python does. For a simple production run:

```bash
pip install gunicorn
gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

Set `HF_TOKEN` and `GEMINI_API_KEY` as real environment variables on
whatever host/VM/PaaS you deploy to (instead of a checked-in `.env`).
Any standard Python host (a VM, Render, Railway, EC2, etc.) works — just
run the command above behind your usual reverse proxy / process manager.

## 7. Limits / notes

- `MAX_PAGES` (default 20) caps how many pages are OCR'd per PDF.
- Large PDFs take a while — each page is a separate OCR API call.
- Gemini is instructed to return `null` for any figure it truly can't find
  rather than guessing, and to log assumptions in the `notes` field.
