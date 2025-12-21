"""Backward-compatible Vertex script. Prefer: python -m fitme vertex ..."""

import sys

from fitme.engines import vertex as vertex_engine


def main() -> None:
    if len(sys.argv) == 1:
        vertex_engine.test_connection()
    elif len(sys.argv) >= 3:
        sample_count = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        vertex_engine.run_tryon(sys.argv[1], sys.argv[2], sample_count)
    else:
        print("Usage:")
        print("  python test_vertex_vto.py")
        print("  python test_vertex_vto.py <person.jpg> <cloth.jpg> [sample_count]")
        sys.exit(1)


if __name__ == "__main__":
    main()
