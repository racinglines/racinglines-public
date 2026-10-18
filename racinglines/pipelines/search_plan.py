"""Code-only F1 sweep planning. No database, sweep execution or scoring."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from racinglines.pipelines.sweep_settings import Settings


def new_candidates(candidates: Iterable[Mapping[str, object]], stored_keys: Iterable[str]) -> list[dict]:
    """Normalize settings and keep the first occurrence of each key absent from this year's stored keys."""
    seen = set(stored_keys)
    out = []
    for candidate in candidates:
        settings = Settings.from_dict(candidate)
        key = settings.key
        if key not in seen:
            seen.add(key)
            out.append(settings.to_json())
    return out


def _count(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def plan(candidates: Iterable[Mapping[str, object]], stored_keys: Iterable[str], *, budget: int = 0,
         history: Iterable[Mapping[str, object]] = (), best_score: float | None = None,
         stop_after: int = 0, margin: float = 0.0) -> dict:
    """Plan at most budget new settings, in input order.

    Scores are supplied outcomes, higher is better; no metric is selected here. History is ordered,
    unique completed settings keys from the same year. The first K misses latch the stop. A strict
    improvement greater than margin resets the streak; even a smaller gain updates the running best.
    Pending candidates have no score and never count as misses.
    """
    _count(budget, "budget")
    _count(stop_after, "stop_after")
    _score(margin, "margin")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if best_score is not None:
        _score(best_score, "best_score")
    if stop_after and best_score is None:
        raise ValueError("best_score is required when stop_after is enabled")
    stored = set(stored_keys)
    if any(not isinstance(key, str) or not key for key in stored):
        raise ValueError("stored_keys must contain non-empty strings")
    completed = []
    seen = set()
    for entry in history:
        if set(entry) != {"settings_key", "score"}:
            raise ValueError("each history entry must contain settings_key and score only")
        key = entry["settings_key"]
        if not isinstance(key, str) or key not in stored:
            raise ValueError("history settings_key must be in this year's stored_keys")
        if key in seen:
            raise ValueError(f"duplicate history settings_key: {key}")
        seen.add(key)
        completed.append(_score(entry["score"], "history score"))
    fresh = new_candidates(candidates, stored)
    misses = 0
    stopped = False
    for score in completed:
        improves = best_score is None or score - best_score > margin
        misses = 0 if improves else misses + 1
        best_score = score if best_score is None else max(best_score, score)
        if stop_after and misses >= stop_after:
            stopped = True
            break
    selected = [] if stopped else fresh[:budget]
    return dict(budget=budget, new_count=len(fresh), selected_count=len(selected),
                stop_after=stop_after, margin=margin, best_score=best_score, consecutive_misses=misses,
                stopped=stopped, reason="stop_rule" if stopped else
                "budget" if len(fresh) > budget else "exhausted",
                candidates=[dict(settings_key=Settings.from_dict(s).key, settings=s) for s in selected])


def print_plan(path: str | Path, *, budget: int = 0, stop_after: int = 0, margin: float = 0.0,
               best_score: float | None = None, dry_run: bool = False) -> None:
    """Read a caller-supplied snapshot; print JSON to stdout only, even without --dry-run."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) - {"year", "stored_keys", "candidates", "history"}:
        raise ValueError("snapshot must be an object containing year, stored_keys, candidates and optional history")
    if type(data.get("year")) is not int or data["year"] <= 0:
        raise ValueError("snapshot year must be a positive integer")
    for name in ("stored_keys", "candidates", "history"):
        value = data.get(name, [] if name == "history" else None)
        if not isinstance(value, list):
            raise ValueError(f"snapshot {name} must be a list")
        if name != "stored_keys" and any(not isinstance(entry, dict) for entry in value):
            raise ValueError(f"snapshot {name} entries must be objects")
    if any(not isinstance(key, str) or not key for key in data["stored_keys"]):
        raise ValueError("snapshot stored_keys must contain non-empty strings")
    result = plan(data["candidates"], data["stored_keys"], budget=budget, history=data.get("history", []),
                  stop_after=stop_after, margin=margin, best_score=best_score)
    print(json.dumps(dict(year=data["year"], dry_run=dry_run, execution_enabled=False, **result),
                     indent=2, sort_keys=True, allow_nan=False))
