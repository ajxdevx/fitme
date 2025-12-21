# FitMe

**Author:** Anass Jid  
**License:** MIT (see [LICENSE](LICENSE))

FitMe is an AI virtual try-on toolkit. Upload a person photo and a clothing image (or copy an outfit from another photo) and generate a realistic try-on result via Google Gemini or Vertex AI.

![Sample try-on output](assets/readme/fitme_tryon_sample.jpg)

Reference pose examples (white background, for input planning):

| One model, six poses | Six people, varied poses |
| --- | --- |
| ![One model poses](assets/readme/fitme_one_model_six_poses.jpg) | ![Six people poses](assets/readme/fitme_six_people_poses.jpg) |

## Project layout

```
fitme/
├── fitme/                 # Application package
│   ├── cli.py             # CLI entry (fitme gemini | fitme vertex)
│   ├── config.py          # Environment and paths
│   ├── prompts.py         # Try-on prompts
│   ├── images.py          # Image load/save helpers
│   ├── preprocess.py      # Optional MediaPipe pipeline
│   └── engines/
│       ├── gemini.py      # Gemini image model
│       └── vertex.py      # Vertex Virtual Try-On API
├── test_data/             # Sample person & clothing images
├── test_results/fitme/    # Generated outputs (gitignored)
├── .env.example           # Copy to .env
└── pyproject.toml
```

## Quick start

### 1. Python environment

```powershell
cd c:\Users\Ajxfu\metafit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. API key

Copy `.env.example` to `.env` and set:

```env
GEMINI_API_KEY=your-key-here
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

The default model is `gemini-3-pro-image-preview`. Override with:

```env
FITME_GEMINI_MODEL=gemini-3-pro-image-preview
```

> **Note:** This image model usually requires **paid** Gemini API quota. If you see `429 RESOURCE_EXHAUSTED`, enable billing in AI Studio.

### 3. Run try-on

**Clothing mode** (product → person):

```powershell
python -m fitme gemini --person test_data/person/woman_standing4.jpg --clothing test_data/clothing/red_dress.jpg
```

**Transfer mode** (outfit from another person):

```powershell
python -m fitme gemini --mode transfer --person test_data/person/woman_standing3.jpg --source test_data/person/woman_standing5.jpg
```

Results: `test_results/fitme/gemini/`

Legacy command (still works):

```powershell
python try_on_test.py --person test_data/person/woman_standing4.jpg --clothing test_data/clothing/red_dress.jpg
```

## Vertex AI (optional)

```powershell
pip install -e ".[vertex]"
```

Set in `.env`:

```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=configs/service-account.json
```

```powershell
python -m fitme vertex --test-connection
python -m fitme vertex --person test_data/person/woman_standing4.jpg --clothing test_data/clothing/red_dress.jpg
```

Results: `test_results/fitme/vertex/`

## Optional preprocessing

```powershell
pip install -e ".[preprocess]"
```

Download MediaPipe task models into `models/` (see `.gitignore`), then:

```powershell
python -m fitme gemini --person ... --clothing ... --preprocess
```

## Third-party services

Try-on generation uses **Google Gemini** and/or **Vertex AI**. Usage is subject to [Google API Terms](https://ai.google.dev/gemini-api/terms) and [Google Cloud Terms](https://cloud.google.com/terms). You are responsible for API keys, billing, and compliance.

## Copyright

Copyright (c) 2026 Anass Jid. All rights reserved under the MIT License.
