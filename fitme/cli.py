"""Command-line interface for FitMe."""

from __future__ import annotations

import argparse
import sys

from fitme.config import DEFAULT_CLOTHING
from fitme.engines import gemini as gemini_engine
from fitme.engines import vertex as vertex_engine
from fitme.prompts import PROMPT_CLOTHING, PROMPT_TRANSFER


def _cmd_gemini(args: argparse.Namespace) -> None:
    if args.mode == "transfer":
        if not args.source:
            print("Error: --source is required for transfer mode")
            sys.exit(1)
        garment = args.source
        prompt = PROMPT_TRANSFER
    else:
        garment = args.clothing or str(DEFAULT_CLOTHING)
        prompt = PROMPT_CLOTHING

    result = gemini_engine.run_tryon(
        args.person,
        garment,
        prompt,
        args.mode,
        use_preprocess=args.preprocess,
    )
    if result:
        print(f"\nTry-on complete! Result: {result}")
    else:
        print("\nNo image generated. Check the model response above.")
        sys.exit(1)


def _cmd_vertex(args: argparse.Namespace) -> None:
    if args.test_connection:
        vertex_engine.test_connection()
        return
    if not args.person or not args.clothing:
        print("Usage: fitme vertex --person <img> --clothing <img>")
        sys.exit(1)
    vertex_engine.run_tryon(args.person, args.clothing, args.samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fitme",
        description="FitMe — AI virtual try-on by Anass Jid",
    )
    sub = parser.add_subparsers(dest="engine", required=True)

    gemini = sub.add_parser("gemini", help="Gemini image model (default engine)")
    gemini.add_argument(
        "--mode",
        default="clothing",
        choices=["clothing", "transfer"],
        help="clothing: product + person; transfer: outfit from source person",
    )
    gemini.add_argument("--person", required=True, help="Target person image")
    gemini.add_argument("--clothing", help="Clothing product image")
    gemini.add_argument("--source", help="Source person wearing clothes (transfer mode)")
    gemini.add_argument(
        "--preprocess",
        action="store_true",
        help="MediaPipe face/pose hints + face restore",
    )
    gemini.set_defaults(func=_cmd_gemini)

    vertex = sub.add_parser("vertex", help="Google Vertex AI Virtual Try-On")
    vertex.add_argument("--person", help="Person image")
    vertex.add_argument("--clothing", help="Product/clothing image")
    vertex.add_argument("--samples", type=int, default=1, help="Number of outputs (1-4)")
    vertex.add_argument(
        "--test-connection",
        action="store_true",
        help="Verify GCP credentials only",
    )
    vertex.set_defaults(func=_cmd_vertex)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
