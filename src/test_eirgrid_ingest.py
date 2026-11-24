"""
Offline tests for eirgrid_ingest.py. They use synthetic payloads shaped like the
API's documented schema, so no network access is needed.

Run with:  python -m unittest test_eirgrid_ingest -v      (or: pytest)

Tests that need pandas are skipped automatically if pandas can't be loaded.
"""
import csv
import gzip
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import eirgrid_ingest as ing

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


def make_times(start: str, periods: int, step_minutes: int = 15):
    t0 = datetime.fromisoformat(start)
    return [(t0 + timedelta(minutes=step_minutes * i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(periods)]


def make_payload(area: str, start: str, periods: int = 96, shift: float = 0.0, drop: set | None = None) -> dict:
    """Build a fake /api/{area} response. `drop` removes point indices to simulate gaps."""
    drop = drop or set()
    step = {"snsp": 30, "frequency": 60}.get(area, 15)
    times = make_times(start, periods, step)

    def pts(fn):
        return [fn(i, t) for i, t in enumerate(times) if i not in drop]

    if area in ("wind", "solar"):
        series = {r: pts(lambda i, t: {"t": t, "actual": 1000.0 + i + shift, "forecast": 900.0 + i}) for r in ("ROI", "NI", "ALL")}
    elif area == "demand":
        series = {"ROI": pts(lambda i, t: {"t": t, "value": 4000.0}),
                  "NI": pts(lambda i, t: {"t": t, "value": 700.0}),
                  "ALL": pts(lambda i, t: {"t": t, "value": 4700.0})}
    elif area == "interconnection":
        series = {c: pts(lambda i, t: {"t": t, "value": 100.0}) for c in ("EWIC", "GRNLK", "MOYLE", "NET")}
    elif area == "co2":
        series = {"ROI": pts(lambda i, t: {"t": t, "value": 500.0}),
                  "NI": pts(lambda i, t: {"t": t, "value": 200.0}),
                  "ALL": pts(lambda i, t: {"t": t, "value": 700.0})}
    elif area == "snsp":
        series = {"ALL": pts(lambda i, t: {"t": t, "value": 60.0})}
    elif area == "frequency":
        series = {"ROI": pts(lambda i, t: {"t": t, "value": 50.0})}
    else:  # generation: snapshots stamped with collection time
        series = {"FUEL_GAS": pts(lambda i, t: {"t": t, "value": 1200.0})}
    return {"generated_at": "2026-03-05T13:23:14Z", "window_days": 30, "series": series}


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._orig = (ing.RAW_DIR, ing.OUT_DIR, ing.fetch_json, ing.REQUEST_PAUSE_S, ing.SOURCE_TZ)
        ing.RAW_DIR = root / "raw"
        ing.OUT_DIR = root / "processed"
        ing.REQUEST_PAUSE_S = 0
        self.calls = {"start": "2026-02-01T00:00:00", "shift": 0.0, "drop": set()}

        def fake_fetch(path, **_):
            if path == "/health":
                return {"status": "ok", "cached_at": "2026-03-05T06:00:00Z", "areas_loaded": ing.AREAS}
            if path == "/api/gaps":
                return {"gaps": []}
            area = path.rsplit("/", 1)[-1]
            return make_payload(area, self.calls["start"], shift=self.calls["shift"], drop=self.calls["drop"])

        ing.fetch_json = fake_fetch

    def tearDown(self):
        ing.RAW_DIR, ing.OUT_DIR, ing.fetch_json, ing.REQUEST_PAUSE_S, ing.SOURCE_TZ = self._orig
        self.tmp.cleanup()


class FetchAndReportTests(BaseCase):
    """These use only the standard library, the same as `fetch` and `report` on your machine."""

    def test_fetch_saves_raw_and_archive(self):
        self.assertEqual(ing.main(["fetch"]), 0)
        for area in ing.AREAS:
            self.assertTrue((ing.OUT_DIR / f"{area}.csv").exists(), area)
            raws = list((ing.RAW_DIR / area).glob("*.json.gz"))
            self.assertTrue(raws, area)
        # raw files are untouched, readable API responses
        raw = next((ing.RAW_DIR / "wind").glob("*.json.gz"))
        with gzip.open(raw, "rt", encoding="utf-8") as fh:
            self.assertIn("series", json.load(fh))
        self.assertTrue(list((ing.RAW_DIR / "gaps").glob("*.json.gz")))

    def test_overlapping_pulls_deduplicate_and_newer_wins(self):
        ing.main(["fetch", "--areas", "wind"])
        first = ing.load_archive_rows("wind")
        # Second pull starts 12 hours later (48 points overlap), with revised actuals.
        self.calls.update(start="2026-02-01T12:00:00", shift=5.0)
        ing.main(["fetch", "--areas", "wind"])
        merged = ing.load_archive_rows("wind")
        with (ing.OUT_DIR / "wind.csv").open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), len({(r["series"], r["t_raw"]) for r in rows}))  # no duplicates
        self.assertGreater(len(merged), len(first))
        overlap = merged[("ROI", "2026-02-01T12:00:00")]
        self.assertAlmostEqual(float(overlap["actual"]), 1005.0)  # index 0 of the second payload, shifted

    def test_failed_area_returns_nonzero_but_continues(self):
        real = ing.fetch_json

        def flaky(path, **kw):
            if path == "/api/solar":
                raise RuntimeError("boom")
            return real(path, **kw)

        ing.fetch_json = flaky
        self.assertEqual(ing.main(["fetch"]), 1)
        self.assertTrue((ing.OUT_DIR / "wind.csv").exists())
        self.assertFalse((ing.OUT_DIR / "solar.csv").exists())

    def test_report_detects_gaps(self):
        self.calls["drop"] = {10, 11, 12}
        ing.main(["fetch", "--areas", "demand"])
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(ing.main(["report"]), 0)
        line = next(l for l in buf.getvalue().splitlines() if l.startswith("demand") and " ROI" in l)
        self.assertIn("1:00:00", line)  # longest gap = 4 steps of 15 min

    def test_report_with_no_data_returns_nonzero(self):
        self.assertEqual(ing.main(["report"]), 1)


class NoPandasTests(BaseCase):
    """Simulate the blocked-pandas situation: fetch and report must still work."""

    def test_fetch_and_report_work_without_pandas(self):
        with mock.patch.dict(sys.modules, {"pandas": None}):
            self.assertEqual(ing.main(["fetch"]), 0)
            self.assertEqual(ing.main(["report"]), 0)
            self.assertEqual(ing.main(["build"]), 1)  # friendly failure, not a crash
        self.assertTrue((ing.OUT_DIR / "demand.csv").exists())


@unittest.skipUnless(HAVE_PANDAS, "pandas not available")
class BuildTests(BaseCase):
    def test_wide_table_columns_and_derived_values(self):
        ing.main(["fetch"])
        wide = ing.build_wide(pd)
        for col in ("wind_actual_ROI", "wind_forecast_ROI", "solar_actual_ROI", "demand_ROI",
                    "ic_NET", "co2_tph_ROI", "snsp_pct_ALL", "wind_error_mw_ROI",
                    "wind_share_of_demand_ROI", "co2_intensity_g_per_kwh_approx_ROI",
                    "t_local", "hour_local"):
            self.assertIn(col, wide.columns)
        self.assertNotIn("generation_FUEL_GAS", wide.columns)
        # 500 t/h over 4000 MW = 0.125 t/MWh = 125 g/kWh
        self.assertAlmostEqual(wide["co2_intensity_g_per_kwh_approx_ROI"].dropna().iloc[0], 125.0)
        # actual 1000 - forecast 900 at index 0
        self.assertAlmostEqual(wide["wind_error_mw_ROI"].dropna().iloc[0], 100.0)

    def test_gaps_appear_as_missing_rows_in_wide_table(self):
        self.calls["drop"] = {10, 11, 12}
        ing.main(["fetch", "--areas", "demand"])
        wide = ing.build_wide(pd)
        self.assertEqual(int(wide["demand_ROI"].isna().sum()), 3)

    def test_source_timezone_conversion(self):
        # If the API's naive times are Irish local time, summer 12:00 is 11:00 UTC.
        ing.SOURCE_TZ = "Europe/Dublin"
        self.calls["start"] = "2026-07-01T12:00:00"
        ing.main(["fetch", "--areas", "demand"])
        wide = ing.build_wide(pd)
        self.assertEqual(str(wide.index.min()), "2026-07-01 11:00:00+00:00")

    def test_build_command_writes_file(self):
        ing.main(["fetch"])
        self.assertEqual(ing.main(["build"]), 0)
        self.assertTrue((ing.OUT_DIR / "eirgrid_wide.csv").exists())


if __name__ == "__main__":
    unittest.main()