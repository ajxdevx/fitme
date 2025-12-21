"""Gemini image model try-on engine."""

from __future__ import annotations

import sys
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from fitme.config import GEMINI_API_KEY, GEMINI_MODEL_ID, OUTPUT_GEMINI
from fitme.images import load_image_part, save_gemini_image
from fitme.preprocess import preprocess_images, restore_face


def _require_api_key() -> str:
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)
    return GEMINI_API_KEY


def _print_quota_help(exc: genai_errors.ClientError) -> None:
    print("\n--- API quota / billing ---")
    print("This model often requires paid Gemini API quota (not free tier).")
    print("Enable billing: https://aistudio.google.com/")
    print("Rate limits: https://ai.google.dev/gemini-api/docs/rate-limits")
    print(f"Details: {exc}")


def run_tryon(
    person_path: str | Path,
    garment_path: str | Path,
    prompt: str,
    mode: str,
    *,
    use_preprocess: bool = False,
) -> Path | None:
    """Generate a try-on image via Gemini."""
    person_path = str(person_path)
    garment_path = str(garment_path)

    extra_prompt = ""
    face_crop_path: str | None = None
    if use_preprocess:
        print("--- Preprocessing with MediaPipe ---")
        person_path, garment_path, extra_prompt, face_crop_path = preprocess_images(
            person_path, garment_path,
        )
        print("--- Preprocessing complete ---")

    client = genai.Client(api_key=_require_api_key())

    print(f"Model: {GEMINI_MODEL_ID}")
    print(f"Mode: {mode}")
    print(f"Preprocess: {use_preprocess}")
    print(f"Person: {person_path}")
    print(f"Garment/Source: {garment_path}")
    print("Generating try-on image...")

    contents: list = [prompt + extra_prompt, load_image_part(person_path), load_image_part(garment_path)]
    if face_crop_path:
        contents.append(load_image_part(face_crop_path))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            _print_quota_help(exc)
        else:
            print(f"Gemini API error: {exc}")
        sys.exit(1)

    result_path = save_gemini_image(
        response, OUTPUT_GEMINI, person_path, garment_path, mode,
    )

    if result_path and use_preprocess:
        print("\n--- Post-processing: Face Restore ---")
        restored = restore_face(person_path, str(result_path))
        if restored:
            print(f"Face-restored result: {restored}")

    return result_path
