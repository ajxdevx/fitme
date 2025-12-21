"""Environment and path configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_ID = os.getenv(
    "FITME_GEMINI_MODEL",
    "gemini-3-pro-image-preview",
)

VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
VERTEX_MODEL = "virtual-try-on-001"
VERTEX_CREDENTIALS = Path(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "configs/service-account.json"),
)

OUTPUT_GEMINI = PROJECT_ROOT / "test_results" / "fitme" / "gemini"
OUTPUT_VERTEX = PROJECT_ROOT / "test_results" / "fitme" / "vertex"

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CLOTHING = PROJECT_ROOT / "test_data" / "clothing" / "tshirt_black.png"
