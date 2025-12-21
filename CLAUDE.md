# FitMe — Virtual Try-On

## Overview

FitMe is an AI virtual try-on system. Users provide a full-body photo and clothing (or a dressed model photo); the app generates a realistic image of the person wearing those clothes.

**Author:** Anass Jid  
**License:** MIT

## Tech stack

- **Try-on (primary):** Google Gemini image models (`fitme/engines/gemini.py`)
- **Try-on (products):** Vertex AI Virtual Try-On (`fitme/engines/vertex.py`)
- **Optional:** MediaPipe preprocessing (`fitme/preprocess.py`)
- **Runtime:** Python 3.10+, `google-genai`, Pillow

## Run try-on (CLI)

```bash
python -m fitme gemini --person <person> --clothing <clothing>
python -m fitme gemini --mode transfer --person <target> --source <dressed_model>
```

- Test images: `test_data/person/`, `test_data/clothing/`
- Output: `test_results/fitme/gemini/`
- Env: `GEMINI_API_KEY` in `.env` (see `.env.example`)

## Project structure

```
fitme/           # Python package (config, CLI, engines)
test_data/       # Sample assets
configs/         # GCP service account path (gitignored JSON keys)
test_results/    # Generated images (gitignored)
```

## Coding conventions

- Python: PEP 8, type hints
- Comments in source: English
- Git commit messages: English

## Documentation

- `README.md` — setup and usage
- `.env.example` — environment variables
