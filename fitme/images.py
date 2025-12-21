"""Image I/O for Gemini and local files."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from google.genai import types


def load_image_part(image_path: str | Path) -> types.Part:
    """Load an image file as a Gemini API Part."""
    path = Path(image_path)
    if not path.exists():
        print(f"Error: Image not found: {path}")
        sys.exit(1)

    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(path.suffix.lower(), "image/png")
    data = path.read_bytes()
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def save_gemini_image(
    response,
    output_dir: Path,
    person_path: str | Path,
    clothing_path: str | Path,
    mode: str,
) -> Path | None:
    """Extract and save the first image from a Gemini response."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    person_name = Path(person_path).stem
    clothing_name = Path(clothing_path).stem

    if not response.candidates:
        print("Error: No candidates in model response.")
        return None

    for i, part in enumerate(response.candidates[0].content.parts):
        if part.inline_data is not None:
            ext = "png" if "png" in part.inline_data.mime_type else "jpg"
            filename = f"{timestamp}_{mode}_{person_name}_{clothing_name}_{i}.{ext}"
            output_path = output_dir / filename
            output_path.write_bytes(part.inline_data.data)
            print(f"Saved: {output_path}")
            return output_path
        if part.text:
            print(f"Model response text: {part.text}")

    return None
