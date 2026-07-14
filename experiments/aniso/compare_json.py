#!/usr/bin/env python
"""
Numerically diff two experiment result JSONs.

Usage:
    python compare_json.py GOLDEN.json NEW.json [--rtol 1e-9] [--exact]

Checks, in order:
  1. Key sets match at every level of nesting (module names, measurement names,
     r / lambda / confidence keys). A missing or extra key is a hard failure --
     values being close is meaningless if the structure drifted.
  2. Every numeric leaf matches. Default is np.isclose(rtol); --exact demands
     bit-identical floats, which is what we expect from changes that only move
     *where* work happens rather than *what* is computed.

NaN == NaN is treated as a match (the scripts serialize NaN as null).
Exits nonzero on any mismatch and prints up to --max-report differences.
"""
import argparse
import json
import math
import sys


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def walk(a, b, path, diffs, rtol, atol, exact):
    # Both null / NaN
    if a is None and b is None:
        return

    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a), set(b)
        for k in sorted(ka - kb):
            diffs.append(f"{path}/{k}: KEY ONLY IN GOLDEN")
        for k in sorted(kb - ka):
            diffs.append(f"{path}/{k}: KEY ONLY IN NEW")
        for k in sorted(ka & kb, key=str):
            walk(a[k], b[k], f"{path}/{k}", diffs, rtol, atol, exact)
        return

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: LIST LENGTH {len(a)} (golden) vs {len(b)} (new)")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs, rtol, atol, exact)
        return

    if is_num(a) and is_num(b):
        if math.isnan(float(a)) and math.isnan(float(b)):
            return
        if exact:
            if float(a) != float(b):
                diffs.append(f"{path}: {a!r} != {b!r} (exact)")
        else:
            if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
                diffs.append(f"{path}: {a!r} != {b!r} (rtol={rtol})")
        return

    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("golden")
    p.add_argument("new")
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--exact", action="store_true")
    p.add_argument("--max-report", type=int, default=40)
    args = p.parse_args()

    with open(args.golden) as f:
        golden = json.load(f)
    with open(args.new) as f:
        new = json.load(f)

    diffs = []
    walk(golden, new, "", diffs, args.rtol, args.atol, args.exact)

    mode = "exact" if args.exact else f"rtol={args.rtol}"
    if not diffs:
        print(f"MATCH ({mode}): {args.golden} == {args.new}")
        return 0

    print(f"MISMATCH ({mode}): {len(diffs)} difference(s)")
    for d in diffs[: args.max_report]:
        print("  " + d)
    if len(diffs) > args.max_report:
        print(f"  ... and {len(diffs) - args.max_report} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
