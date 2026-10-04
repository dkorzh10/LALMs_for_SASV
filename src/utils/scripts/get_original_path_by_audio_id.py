#!/usr/bin/env python3
"""
Look up original_path by audio_id from unified_antispoofing_dataset.json.
Supports very large JSON via streaming (ijson). For huge files, install: pip install ijson
"""
import argparse
import os
import json
import sys

DEFAULT_JSON = os.environ.get("UNIFIED_DATASET_JSON", "./data/unified_antispoofing_dataset.json")


def _lookup_with_json_load(path: str, want: set) -> dict:
    """Load full JSON (works for smaller files)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("samples", data.get("data", list(data.values())[0] if data else []))
        if not isinstance(items, list):
            items = []
    else:
        items = []
    found = {}
    for item in items:
        aid = item.get("audio_id")
        if aid in want:
            found[aid] = item.get("original_path", "")
            want = want - {aid}
            if not want:
                break
    return found


def _lookup_with_ijson(path: str, want: set, array_key: str | None) -> dict:
    """Stream JSON array and collect matches (for very large files)."""
    import ijson
    found = {}
    prefix = f"{array_key}.item" if array_key else "item"
    with open(path, "rb") as f:
        for item in ijson.items(f, prefix):
            aid = item.get("audio_id")
            if aid in want:
                found[aid] = item.get("original_path", "")
                want = want - {aid}
            if not want:
                break
    return found


def main():
    parser = argparse.ArgumentParser(
        description="Get original_path by audio_id from unified antispoofing dataset JSON."
    )
    parser.add_argument(
        "audio_id",
        nargs="+",
        help="One or more audio_id(s) to look up.",
    )
    parser.add_argument(
        "--json",
        "-j",
        default=DEFAULT_JSON,
        help="Path to unified_antispoofing_dataset.json",
    )
    parser.add_argument(
        "--array-key",
        default=None,
        help="If JSON root is an object, key of the array (e.g. 'samples'). Default: root is array.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Force streaming with ijson (recommended for very large JSON).",
    )
    args = parser.parse_args()
    want = set(args.audio_id)
    found = {}

    try:
        import ijson  # noqa: F401
        ijson_available = True
    except ImportError:
        ijson_available = False

    use_stream = args.stream or ijson_available
    if args.stream and not ijson_available:
        print("pip install ijson required for --stream", file=sys.stderr)
        sys.exit(2)

    try:
        if use_stream:
            found = _lookup_with_ijson(args.json, want, args.array_key)
        else:
            found = _lookup_with_json_load(args.json, want)
    except (MemoryError, ValueError) as e:
        print(f"JSON too large or invalid: {e}. Run: pip install ijson (then re-run with --stream if needed)", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError:
        print(f"File not found: {args.json}", file=sys.stderr)
        sys.exit(1)

    missing = [aid for aid in args.audio_id if aid not in found]
    if missing:
        for aid in missing:
            print(f"audio_id not found: {aid}", file=sys.stderr)
        sys.exit(1)

    for aid in args.audio_id:
        print(found[aid])


if __name__ == "__main__":
    main()
