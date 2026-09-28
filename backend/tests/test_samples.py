#!/usr/bin/env python3
"""The sample gallery's manifest: every "ready" sample must actually work.

frontend/public/samples/manifest.json drives the console's "Try a sample"
gallery. A sample flips from "missing" to "ready" when its files are added;
these checks make that flip safe. A ready sample must have its files, its
source, licence and attribution, and imagery the engine for its layout can
read:

- before/after: the change engine opens both dates with PIL as RGB, so they
  must be 8-bit, 3- or 4-band, the same size, and — when georeferenced — the
  same CRS and footprint. Dates, when given, run earliest first.
- optical + SAR: 13 Sentinel-2 bands and 2 Sentinel-1 bands, the same size.

    python backend/tests/test_samples.py
    pytest backend/tests
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "frontend" / "public"
MANIFEST = PUBLIC / "samples" / "manifest.json"

#: Mirrors MODE_ROLES in backend/app/routes/upload.py, in slot order.
ROLES = {"single": ["primary"], "bi_temporal": ["t0", "t1"], "cross_modal": ["optical", "sar"]}


def _samples() -> list[dict]:
    return json.loads(MANIFEST.read_text())["samples"]


def _ready(mode: str | None = None) -> list[dict]:
    return [s for s in _samples() if s["status"] == "ready" and (mode is None or s["mode"] == mode)]


def _path(entry: dict) -> Path:
    return PUBLIC / entry["path"].lstrip("/")


def _raster(path: Path):
    import rasterio

    return rasterio.open(path)


def test_manifest_shape():
    samples = _samples()
    ids = [s["id"] for s in samples]
    assert len(ids) == len(set(ids)), f"duplicate sample ids: {ids}"
    for s in samples:
        assert s["status"] in {"ready", "missing"}, s["id"]
        assert s["mode"] in ROLES, s["id"]
        assert s["question"].strip(), f"{s['id']}: no starter question"
        if s["status"] == "missing":
            assert s.get("needs"), f"{s['id']}: a missing sample must say what it needs"


def test_ready_samples_have_files_and_credit():
    for s in _ready():
        roles = [f["role"] for f in s["files"]]
        assert roles == ROLES[s["mode"]], f"{s['id']}: roles {roles}, expected {ROLES[s['mode']]}"
        for entry in s["files"]:
            assert _path(entry).is_file(), f"{s['id']}: missing file {entry['path']}"
        if s.get("thumbnail"):
            assert (PUBLIC / s["thumbnail"].lstrip("/")).is_file(), f"{s['id']}: missing thumbnail"
        for field in ("source", "licence", "attribution"):
            assert s.get(field), f"{s['id']}: {field} is required before it can be ready"
        assert "[year]" not in s["attribution"], f"{s['id']}: fill in the year in the attribution"


def test_single_images_are_readable():
    from PIL import Image

    for s in _ready("single"):
        path = _path(s["files"][0])
        if path.suffix.lower() in {".tif", ".tiff"}:
            with _raster(path) as src:
                assert src.count >= 3, f"{s['id']}: {src.count} band(s)"
        else:
            Image.open(path).convert("RGB")


def test_before_after_pairs_fit_the_change_engine():
    from PIL import Image

    for s in _ready("bi_temporal"):
        before, after = (_path(f) for f in s["files"])
        sizes = []
        for path in (before, after):
            # Exactly what change.py does with each date.
            try:
                with Image.open(path) as img:
                    img.convert("RGB")
                    sizes.append(img.size)
            except Exception as exc:  # noqa: BLE001 - reported as the fix to make
                raise AssertionError(
                    f"{s['id']}: the change engine cannot open {path.name} ({exc}). "
                    "Export it as an 8-bit TIFF or PNG true-colour image."
                ) from exc
            with _raster(path) as src:
                assert src.count in (3, 4), f"{s['id']}: {path.name} has {src.count} bands, need 3"
                assert src.dtypes[0] == "uint8", f"{s['id']}: {path.name} is {src.dtypes[0]}, need 8-bit"
        assert sizes[0] == sizes[1], f"{s['id']}: sizes differ {sizes}"

        with _raster(before) as a, _raster(after) as b:
            if a.crs and b.crs:
                assert a.crs == b.crs, f"{s['id']}: CRS differs ({a.crs} vs {b.crs})"
                tolerance = max(abs(a.res[0]), abs(a.res[1]))
                for x, y in zip(a.bounds, b.bounds):
                    assert abs(x - y) <= tolerance, (
                        f"{s['id']}: footprints differ by more than a pixel: {a.bounds} vs {b.bounds}"
                    )

        dates = s.get("dates") or {}
        if dates.get("t0") and dates.get("t1"):
            assert date.fromisoformat(dates["t0"]) < date.fromisoformat(dates["t1"]), (
                f"{s['id']}: the before date must come first"
            )
        assert "Copernicus Sentinel data" in s["attribution"] or "Copernicus" not in s["source"]["name"]


def test_optical_sar_pairs_fit_the_fusion_encoders():
    for s in _ready("cross_modal"):
        optical, sar = (_path(f) for f in s["files"])
        with _raster(optical) as o, _raster(sar) as r:
            assert o.count == 13, f"{s['id']}: optical has {o.count} bands, the encoder needs 13"
            assert r.count == 2, f"{s['id']}: SAR has {r.count} bands, the encoder needs 2 (VV, VH)"
            assert (o.width, o.height) == (r.width, r.height), f"{s['id']}: sizes differ"


def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a runner reports, it does not raise
            failures.append(name)
            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS  {name}")
    print(f"\n{len(tests) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
