#!/usr/bin/env python3
"""
eirgrid_ingest.py: archive and tidy EirGrid Smart Grid Dashboard data
pulled from the third-party "EirGrid Data API" (https://grid.18for0.ie).

WHY THIS EXISTS
    The API only serves the trailing ~30 days. To get a long record (winter
    peaks, windy nights, several seasons) you have to keep your own archive.

    Every pull returns the full trailing 30 days, so pulls overlap. Nothing is
    lost as long as you pull again within ~25 days. Weekly is plenty; daily
    scheduling is just a convenience.

USAGE
    python eirgrid_ingest.py fetch            # pull all areas, save raw JSON, merge into archive
    python eirgrid_ingest.py fetch --areas wind demand
    python eirgrid_ingest.py report           # coverage and gap summary of your archive
    python eirgrid_ingest.py build            # build one 15-minute analysis table (needs pandas)

DEPENDENCIES
    `fetch` and `report` use only the Python standard library, so they run even
    where compiled packages (pandas) are blocked. Only `build` needs pandas; if
    pandas can't load on your machine, run `build` on Google Colab or a GitHub
    Codespace with the data/processed folder.

OUTPUT
    data/raw/eirgrid/<area>/<area>_<UTC retrieval time>.json.gz   (untouched API responses, never edit)
    data/processed/<area>.csv                                      (de-duplicated archive per area; times exactly as the API gave them)
    data/processed/eirgrid_wide.csv                                (15-minute analysis table from `build`)

THINGS YOU MUST VERIFY YOURSELF (see the checklist printed by `report`)
    1. SOURCE_TZ below: whether the API's naive timestamps are UTC or Irish local time.
    2. That values match the official Smart Grid Dashboard for a few timestamps.
    3. The API operator's terms/licence (contact link is in /openapi.json).
    4. Cite EirGrid's Smart Grid Dashboard as the original source in your report.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_URL = "https://grid.18for0.ie"
AREAS = ["wind", "solar", "demand", "interconnection", "co2", "frequency", "snsp", "generation"]

# Areas that go into the analysis table. `frequency` (hourly average, ROI only) and
# `generation` (rolling 24 h fuel energy stamped with collection time, not effective
# time) are archived but left out of the wide table because they don't suit this project.
WIDE_AREAS = ["wind", "solar", "demand", "interconnection", "co2", "snsp"]

# The API's "t" values look like "2026-02-03T13:30:00" with no timezone marker.
# TODO: verify against the official dashboard, then set to "UTC" or "Europe/Dublin".
# Only `build` uses this; the archive keeps the times exactly as the API gave them.
SOURCE_TZ = "UTC"
LOCAL_TZ = "Europe/Dublin"

RAW_DIR = Path("data/raw/eirgrid")
OUT_DIR = Path("data/processed")

USER_AGENT = "cleanergrid-student-project/0.1 (add your email here)"
REQUEST_PAUSE_S = 1.0  # be polite to a free, third-party service

ARCHIVE_COLUMNS = ["t_raw", "series", "actual", "forecast", "value", "retrieved_at", "source_generated_at"]


# ----------------------------------------------------------------------------
# Fetching (standard library only)
# ----------------------------------------------------------------------------
def fetch_json(path: str, retries: int = 3, timeout: int = 30) -> dict:
    """GET a path from the API with simple retry/backoff. Returns parsed JSON."""
    url = f"{BASE_URL}{path}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError) as exc:  # URLError, HTTPError, timeouts, bad JSON
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_error}")


def save_raw(name: str, payload: dict, retrieved: datetime) -> Path:
    """Store the untouched API response (gzipped) so results can always be re-derived."""
    folder = RAW_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}_{retrieved.strftime('%Y%m%dT%H%M%SZ')}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


# ----------------------------------------------------------------------------
# Parsing and archiving (standard library only)
# ----------------------------------------------------------------------------
def parse_area_rows(payload: dict, retrieved_at: str) -> list[dict]:
    """Flatten one /api/{area} response into rows (one per timestamp per series)."""
    rows = []
    generated = payload.get("generated_at") or ""
    for series_key, points in (payload.get("series") or {}).items():
        for point in points:
            t = point.get("t")
            if not t:
                continue
            rows.append({
                "t_raw": str(t),
                "series": series_key,
                "actual": _cell(point.get("actual")),
                "forecast": _cell(point.get("forecast")),
                "value": _cell(point.get("value")),
                "retrieved_at": retrieved_at,
                "source_generated_at": generated,
            })
    return rows


def _cell(x) -> str:
    return "" if x is None else str(x)


def load_archive_rows(area: str) -> dict[tuple[str, str], dict]:
    """Read data/processed/<area>.csv into {(series, t_raw): row}."""
    path = OUT_DIR / f"{area}.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {(r["series"], r["t_raw"]): r for r in csv.DictReader(fh)}


def merge_into_archive(area: str, new_rows: list[dict]) -> int:
    """Merge rows into data/processed/<area>.csv. Newer pulls overwrite older ones for the
    same (series, timestamp), so later revisions of the data win. Returns rows added."""
    if not new_rows:
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    archive = load_archive_rows(area)
    before = len(archive)
    for row in new_rows:
        archive[(row["series"], row["t_raw"])] = row

    path = OUT_DIR / f"{area}.csv"
    tmp = path.with_suffix(".csv.tmp")  # write-then-replace so an interrupted run can't corrupt the archive
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ARCHIVE_COLUMNS)
        writer.writeheader()
        for key in sorted(archive):
            writer.writerow(archive[key])
    tmp.replace(path)
    return len(archive) - before


def cmd_fetch(args: argparse.Namespace) -> int:
    retrieved = datetime.now(timezone.utc)
    retrieved_iso = retrieved.isoformat(timespec="seconds")
    failures = 0

    try:
        health = fetch_json("/health")
        print(f"API status: {health.get('status')}, cache last refreshed: {health.get('cached_at')}")
    except RuntimeError as exc:
        print(f"[WARN] health check failed: {exc}")

    for area in args.areas or AREAS:
        try:
            payload = fetch_json(f"/api/{area}")
        except RuntimeError as exc:
            print(f"[WARN] {exc}")
            failures += 1
            continue
        raw_path = save_raw(area, payload, retrieved)
        rows = parse_area_rows(payload, retrieved_iso)
        added = merge_into_archive(area, rows)
        print(f"{area:<16} {len(rows):>7} rows pulled, {added:>6} new  (raw: {raw_path})")
        time.sleep(REQUEST_PAUSE_S)

    # Keep the operator's own gap report as provenance for data quality notes.
    try:
        save_raw("gaps", fetch_json("/api/gaps"), retrieved)
    except RuntimeError as exc:
        print(f"[WARN] gap report not saved: {exc}")

    return 1 if failures else 0


# ----------------------------------------------------------------------------
# Coverage report (standard library only)
# ----------------------------------------------------------------------------
def _parse_t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def cmd_report(_: argparse.Namespace) -> int:
    print(f"{'area':<16}{'series':<10}{'rows':>8}  {'first':<20}{'last':<20}{'missing %':>10}  {'longest gap':>12}")
    any_data = False
    for area in AREAS:
        rows = load_archive_rows(area)
        if not rows:
            continue
        any_data = True
        by_series: dict[str, list[datetime]] = {}
        for (series, t_raw) in rows:
            by_series.setdefault(series, []).append(_parse_t(t_raw))
        for series, times in sorted(by_series.items()):
            times.sort()
            first, last = str(times[0])[:19], str(times[-1])[:19]
            if area == "generation":  # snapshots stamped with collection time; no regular cadence
                print(f"{area:<16}{series:<10}{len(times):>8}  {first:<20}{last:<20}{'n/a':>10}  {'n/a':>12}")
                continue
            steps = [b - a for a, b in zip(times, times[1:])]
            if steps:
                step = Counter(steps).most_common(1)[0][0]
                expected = int((times[-1] - times[0]) / step) + 1
                missing_pct = 100 * (expected - len(times)) / max(expected, 1)
                longest = max(steps)
            else:
                missing_pct, longest = 0.0, 0
            print(f"{area:<16}{series:<10}{len(times):>8}  {first:<20}{last:<20}{missing_pct:>9.1f}%  {str(longest):>12}")
    if not any_data:
        print("Nothing archived yet. Run `fetch` first.")
        return 1

    print(
        "\nBefore you rely on this data:\n"
        "  [ ] Confirm SOURCE_TZ by comparing a few timestamps with the official Smart Grid Dashboard\n"
        "  [ ] Spot-check values (demand, wind, SNSP) against the official dashboard\n"
        "  [ ] Check the API operator's terms/licence (contact link in /openapi.json)\n"
        "  [ ] Confirm the unit of the co2 series (the API says tonnes/hour) on the official dashboard\n"
        "  [ ] Cite EirGrid's Smart Grid Dashboard as the original source, with your retrieval dates"
    )
    return 0


# ----------------------------------------------------------------------------
# Analysis table (needs pandas; imported lazily so fetch/report never depend on it)
# ----------------------------------------------------------------------------
def load_archive_df(pd, area: str):
    path = OUT_DIR / f"{area}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["t_raw"], errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(SOURCE_TZ, ambiguous="NaT", nonexistent="NaT")
    df["t_utc"] = ts.dt.tz_convert("UTC")
    df = df.dropna(subset=["t_utc"])
    return df.drop_duplicates(subset=["t_utc", "series"], keep="last")


def pivot_area(pd, area: str):
    """Archive -> wide columns named <area>_<field>_<series>, indexed by UTC time."""
    df = load_archive_df(pd, area)
    if df.empty:
        return pd.DataFrame()
    pieces = []
    if area in ("wind", "solar"):
        for field in ("actual", "forecast"):
            p = df.pivot(index="t_utc", columns="series", values=field)
            p.columns = [f"{area}_{field}_{c}" for c in p.columns]
            pieces.append(p)
    else:
        prefix = {"interconnection": "ic", "co2": "co2_tph", "snsp": "snsp_pct"}.get(area, area)
        p = df.pivot(index="t_utc", columns="series", values="value")
        p.columns = [f"{prefix}_{c}" for c in p.columns]
        pieces.append(p)
    return pd.concat(pieces, axis=1)


def build_wide(pd):
    frames = [f for f in (pivot_area(pd, a) for a in WIDE_AREAS) if not f.empty]
    if not frames:
        raise SystemExit("No archived data found. Run `fetch` first.")
    wide = pd.concat(frames, axis=1).sort_index()

    # Put everything on a regular 15-minute grid so gaps show up as NaN rather than vanishing.
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="15min", tz="UTC")
    wide = wide.reindex(grid)
    wide.index.name = "t_utc"

    # Calendar helpers for defining events (local Irish time).
    local = wide.index.tz_convert(LOCAL_TZ)
    wide.insert(0, "dow_local", local.dayofweek)
    wide.insert(0, "hour_local", local.hour + local.minute / 60)
    wide.insert(0, "t_local", local.strftime("%Y-%m-%d %H:%M"))

    # Derived columns. These are PROXIES, not official EirGrid metrics.
    for region in ("ROI", "NI", "ALL"):
        a, f = f"wind_actual_{region}", f"wind_forecast_{region}"
        if a in wide and f in wide:
            wide[f"wind_error_mw_{region}"] = wide[a] - wide[f]
        d = f"demand_{region}"
        if a in wide and d in wide:
            wide[f"wind_share_of_demand_{region}"] = wide[a] / wide[d]
        c = f"co2_tph_{region}"
        if c in wide and d in wide:
            # tonnes CO2 per hour / MW of demand = t/MWh; x1000 = g/kWh. Check the unit
            # of co2 against the official dashboard before using this in the report.
            wide[f"co2_intensity_g_per_kwh_approx_{region}"] = wide[c] / wide[d] * 1000
    return wide


def cmd_build(_: argparse.Namespace) -> int:
    try:
        import pandas as pd
    except ImportError as exc:
        print(
            f"`build` needs pandas, which can't be loaded here ({exc}).\n"
            "Your archive in data/processed is unaffected. Run `build` somewhere pandas works\n"
            "(Google Colab, a GitHub Codespace, or another machine) using the same folder."
        )
        return 1
    wide = build_wide(pd)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "eirgrid_wide.csv"
    wide.to_csv(path)
    print(f"Wrote {path}: {len(wide)} rows x {wide.shape[1]} columns")
    print(f"Range (UTC): {wide.index.min()} to {wide.index.max()}")
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive and tidy EirGrid dashboard data.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download all areas and merge into the archive")
    p_fetch.add_argument("--areas", nargs="+", choices=AREAS, help="limit to specific areas")
    p_fetch.set_defaults(func=cmd_fetch)

    p_report = sub.add_parser("report", help="coverage and gap summary of the archive")
    p_report.set_defaults(func=cmd_report)

    p_build = sub.add_parser("build", help="build the 15-minute wide analysis table (needs pandas)")
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())