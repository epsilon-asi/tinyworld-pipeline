"""Export committed JSON schemas (``docs/schemas/*.schema.json``).

Run after changing a contract; CI's drift test fails when the committed file is stale::

    uv run python tools/export_schemas.py
    uv run python tools/export_schemas.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tinyworld_pipeline.domain.manifest import manifest_json_schema

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = ROOT / "docs" / "schemas"


def render_all() -> dict[str, str]:
    """``{filename: json_text}`` for every committed schema."""
    schemas = {"manifest.schema.json": manifest_json_schema()}
    return {
        name: json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        for name, schema in schemas.items()
    }


def main(argv: list[str] | None = None) -> int:
    """Write (or with ``--check`` compare) the schema files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if files differ")
    args = parser.parse_args(argv)
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for name, text in render_all().items():
        path = SCHEMAS_DIR / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        print(f"stale schemas: {stale}; run tools/export_schemas.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
