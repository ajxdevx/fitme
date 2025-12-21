"""Backward-compatible entry point. Prefer: python -m fitme gemini ..."""

from fitme.cli import main

if __name__ == "__main__":
    import sys

    # Map legacy: try_on_test.py --person x --clothing y
    if len(sys.argv) > 1 and sys.argv[1] not in ("gemini", "vertex", "-h", "--help"):
        sys.argv.insert(1, "gemini")
    main()
