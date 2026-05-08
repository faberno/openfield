from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "validation" / "results"


def load_cases() -> list[dict]:
    return json.loads((ROOT / "validation" / "cases.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Field II and openfield validation outputs.")
    parser.add_argument("cases", nargs="*", help="Case names to compare.")
    parser.add_argument("--all", action="store_true", help="Compare all cases in cases.json.")
    parser.add_argument("--rtol", type=float, default=None, help="Override relative tolerance.")
    parser.add_argument("--atol", type=float, default=None, help="Override absolute tolerance.")
    parser.add_argument("--strict", action="store_true", help="Treat expected failures as failures.")
    args = parser.parse_args()

    cases = load_cases()
    by_name = {case["name"]: case for case in cases}
    selected = list(by_name) if args.all or not args.cases else args.cases
    unknown = sorted(set(selected) - set(by_name))
    if unknown:
        raise SystemExit(f"Unknown validation case(s): {', '.join(unknown)}")

    failed = []
    for name in selected:
        case = by_name[name]
        rtol = case["rtol"] if args.rtol is None else args.rtol
        atol = case["atol"] if args.atol is None else args.atol
        expected_failure = bool(case.get("expected_failure", False))
        try:
            compare_case(name, rtol=rtol, atol=atol)
        except AssertionError as exc:
            if expected_failure and not args.strict:
                reason = case.get("reason", "expected failure")
                print(f"XFAIL {name}: {exc} ({reason})")
            else:
                failed.append((name, str(exc)))
                print(f"FAIL {name}: {exc}")
        else:
            if expected_failure and not args.strict:
                print(f"XPASS {name}: expected failure no longer fails")
            else:
                print(f"PASS {name}")

    if failed:
        raise SystemExit(1)


def compare_case(case_name: str, *, rtol: float, atol: float) -> None:
    fieldii_dir = RESULTS / case_name / "fieldii"
    openfield_dir = RESULTS / case_name / "openfield"
    if not fieldii_dir.exists():
        raise AssertionError(f"missing Field II output directory: {fieldii_dir}")
    if not openfield_dir.exists():
        raise AssertionError(f"missing openfield output directory: {openfield_dir}")

    fieldii_files = {path.name for path in fieldii_dir.glob("*.csv")}
    openfield_files = {path.name for path in openfield_dir.glob("*.csv")}
    common_files = sorted(fieldii_files & openfield_files)
    if not common_files:
        raise AssertionError("no common CSV files to compare")
    missing = sorted(fieldii_files ^ openfield_files)
    if missing:
        raise AssertionError(f"mismatched CSV file set: {missing}")

    for filename in common_files:
        expected = load_csv(fieldii_dir / filename)
        actual = load_csv(openfield_dir / filename)
        if expected.shape != actual.shape:
            raise AssertionError(f"{filename}: shape mismatch {expected.shape} != {actual.shape}")
        if not np.allclose(expected, actual, rtol=rtol, atol=atol):
            diff = np.abs(expected - actual)
            raise AssertionError(
                f"{filename}: max abs diff {float(np.max(diff)):.6g} exceeds rtol={rtol}, atol={atol}"
            )


def load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    return np.asarray(data, dtype=np.float64)


if __name__ == "__main__":
    main()
