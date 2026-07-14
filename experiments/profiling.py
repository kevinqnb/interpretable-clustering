"""
Lightweight, opt-in profiling helpers for the experiment scripts.

Design goals:
- Zero overhead and zero behavior change when profiling is disabled (the default).
- Survives joblib/loky worker dispatch: per-module timings are returned inside the
  result dict (see experiment._run_fit) rather than relying on shared process state.
- Human-readable end-of-run report so we can rank stages by wall-clock.

Typical use in a script:

    from experiments.profiling import StageTimer
    timer = StageTimer(enabled=True)
    with timer.stage("mine-load"):
        ...
    with timer.stage("ids-cache-build"):
        ...
    timer.report()

For per-module fit-vs-measurement splits, pass `profile=True` to Experiment(...);
the breakdown is attached to result_dict['_profile'] and printed by exp.run().
"""

import time
from contextlib import contextmanager


####################################################################################################
# Minimal import-and-go stage stamps for the experiment scripts.
#
# Usage (already wired into max_rules.py / confidence.py / lambda.py):
#     from experiments.profiling import stamp, stamp_reset
#     stamp_reset()
#     ...
#     stamp("data loaded")
#     ...
#     stamp("exp.run: module fits")
#
# Each stamp prints the time since the previous stamp and the cumulative total, so a
# run's console log shows exactly how long each stage took. Lines are prefixed with
# [TIMING] for easy grepping (e.g. `python lambda.py | tee run.log; grep TIMING run.log`).

_STAMP_START = None
_STAMP_LAST = None


def stamp_reset() -> None:
    """(Re)start the stage clock. Call once near the top of a script."""
    global _STAMP_START, _STAMP_LAST
    _STAMP_START = _STAMP_LAST = time.perf_counter()


def stamp(label: str = "") -> None:
    """Print elapsed time since the previous stamp() and the cumulative total."""
    global _STAMP_START, _STAMP_LAST
    now = time.perf_counter()
    if _STAMP_START is None:
        _STAMP_START = _STAMP_LAST = now
    delta = now - _STAMP_LAST
    total = now - _STAMP_START
    print(f"[TIMING] {label:<52} +{delta:9.2f}s   (total {total / 60:6.2f} min)", flush=True)
    _STAMP_LAST = now


####################################################################################################


class StageTimer:
    """Accumulates named stage durations and prints a ranked report."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        # name -> [total_seconds, n_calls]
        self._stages: dict[str, list] = {}

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - start
            entry = self._stages.setdefault(name, [0.0, 0])
            entry[0] += dt
            entry[1] += 1

    def add(self, name: str, seconds: float, n_calls: int = 1):
        """Manually fold in a duration (e.g. one returned from a worker process)."""
        if not self.enabled:
            return
        entry = self._stages.setdefault(name, [0.0, 0])
        entry[0] += seconds
        entry[1] += n_calls

    def merge(self, other: dict):
        """Fold in another {name: [seconds, n_calls]} mapping (e.g. from a worker)."""
        if not self.enabled or not other:
            return
        for name, (secs, n) in other.items():
            self.add(name, secs, n)

    def as_dict(self) -> dict:
        return {name: [round(secs, 6), n] for name, (secs, n) in self._stages.items()}

    def report(self, title: str = "PROFILE"):
        if not self.enabled or not self._stages:
            return
        rows = sorted(self._stages.items(), key=lambda kv: kv[1][0], reverse=True)
        total = sum(secs for _, (secs, _) in rows)
        width = max((len(name) for name, _ in rows), default=10)
        print()
        print("=" * (width + 34))
        print(f"{title} (total tracked: {total:.2f}s)")
        print("=" * (width + 34))
        print(f"{'stage'.ljust(width)}   {'seconds':>10}  {'calls':>7}  {'s/call':>9}")
        print("-" * (width + 34))
        for name, (secs, n) in rows:
            per = secs / n if n else 0.0
            print(f"{name.ljust(width)}   {secs:>10.3f}  {n:>7d}  {per:>9.4f}")
        print("=" * (width + 34))
        print()
