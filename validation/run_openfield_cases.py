from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "validation" / "cases"


def load_cases() -> list[dict]:
    return json.loads((ROOT / "validation" / "cases.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run openfield validation cases.")
    parser.add_argument("cases", nargs="*", help="Case names to run. Defaults to all cases.")
    parser.add_argument("--list", action="store_true", help="List available cases and exit.")
    args = parser.parse_args()

    cases = load_cases()
    names = [case["name"] for case in cases]
    if args.list:
        for name in names:
            print(name)
        return

    selected = args.cases or names
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise SystemExit(f"Unknown validation case(s): {', '.join(unknown)}")

    for name in selected:
        script = CASE_ROOT / name / "openfield_case.py"
        print(f"Running openfield validation case: {name}")
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
