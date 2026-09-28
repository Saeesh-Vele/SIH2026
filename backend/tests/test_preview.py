#!/usr/bin/env python3
"""Upload metadata and image previews: honest georeferencing, owner-only access.

- A georeferenced GeoTIFF reports its CRS, centre and pixel size, read from
  its own header; a PNG reports that it has none.
- The preview endpoint renders the true-colour picture the models see, and
  only for the upload's owner.
- Upload ids that are not the server's own hex ids never touch the disk.

Same rules as the suites beside it: in-memory Mongo, no network, no GPU. The
fixtures (fake verifier, fake database) come from test_auth.

    python backend/tests/test_preview.py
    pytest backend/tests
"""

from __future__ import annotations

import io
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_auth import bearer, client, isolated  # noqa: E402


def _png_bytes(size: int = 16) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (40, 120, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def _sentinel2_geotiff() -> bytes:
    """A 13-band, 64 px tile in UTM 33N with 10 m pixels, like EuroSAT MS."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    path = Path(tempfile.mkdtemp()) / "s2.tif"
    data = (np.random.default_rng(0).random((13, 64, 64)) * 3000).astype("uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=13,
        dtype="uint16",
        crs="EPSG:32633",
        transform=from_origin(500000, 5000000, 10, 10),
    ) as dst:
        dst.write(data)
    return path.read_bytes()


def _upload(token: str, name: str, payload: bytes, benchmark: bool) -> dict:
    response = client().post(
        "/api/upload",
        data={"mode": "single", "benchmark_mode": str(benchmark).lower()},
        files={"files": (name, io.BytesIO(payload))},
        headers=bearer(token),
    )
    assert response.status_code == 201, response.text
    return response.json()


@isolated
def test_geotiff_reports_its_own_georeferencing():
    body = _upload("token-alice", "scene.tif", _sentinel2_geotiff(), benchmark=False)
    geo = body["assets"][0]["geo"]
    assert geo["status"] == "georeferenced", geo
    assert geo["crs"] == "EPSG:32633"
    assert geo["bands"] == 13 and geo["width"] == 64
    assert geo["gsd_m"] == 10.0
    # UTM 33N, easting 500 km, northing ~5000 km: on the 15°E meridian, ~45°N.
    assert abs(geo["lon"] - 15.0) < 0.05, geo
    assert 44.9 < geo["lat"] < 45.2, geo


@isolated
def test_png_says_it_has_no_georeferencing():
    body = _upload("token-alice", "tile.png", _png_bytes(), benchmark=True)
    geo = body["assets"][0]["geo"]
    assert geo["status"] == "none", geo
    assert geo["lat"] is None and geo["crs"] is None


@isolated
def test_preview_is_a_true_colour_png_for_the_owner():
    from PIL import Image

    body = _upload("token-alice", "scene.tif", _sentinel2_geotiff(), benchmark=False)
    upload_id, asset_id = body["upload_id"], body["assets"][0]["asset_id"]
    url = f"/api/upload/{upload_id}/assets/{asset_id}/preview"

    mine = client().get(url, headers=bearer("token-alice"))
    assert mine.status_code == 200, mine.text
    assert mine.headers["content-type"] == "image/png"
    picture = Image.open(io.BytesIO(mine.content))
    assert picture.mode == "RGB" and picture.size == (64, 64)

    # Cached beside the upload, and served the same the second time.
    again = client().get(url, headers=bearer("token-alice"))
    assert again.content == mine.content

    assert client().get(url, headers=bearer("token-bob")).status_code == 404
    assert client().get(url).status_code == 401


@isolated
def test_thumbnail_is_the_first_image_for_the_owner():
    body = _upload("token-alice", "tile.png", _png_bytes(), benchmark=True)
    url = f"/api/upload/{body['upload_id']}/thumbnail"
    mine = client().get(url, headers=bearer("token-alice"))
    assert mine.status_code == 200 and mine.headers["content-type"] == "image/png", mine.text
    assert client().get(url, headers=bearer("token-bob")).status_code == 404


@isolated
def test_get_upload_is_owner_only():
    body = _upload("token-alice", "tile.png", _png_bytes(), benchmark=True)
    url = f"/api/upload/{body['upload_id']}"

    mine = client().get(url, headers=bearer("token-alice"))
    assert mine.status_code == 200, mine.text
    assert mine.json()["assets"][0]["filename"] == "tile.png"
    assert client().get(url, headers=bearer("token-bob")).status_code == 404


@isolated
def test_malformed_upload_ids_are_not_found():
    for bad in ("..", "abc", "0" * 31, "Z" * 32):
        response = client().get(f"/api/upload/{bad}", headers=bearer("token-alice"))
        assert response.status_code == 404, (bad, response.status_code)
    body = client().post(
        "/api/query",
        json={"query": "Describe this scene.", "upload_id": "../../etc"},
        headers=bearer("token-alice"),
    ).json()
    assert body["status"] == "rejected", body


@isolated
def test_missing_file_is_gone_not_a_crash():
    body = _upload("token-alice", "tile.png", _png_bytes(), benchmark=True)
    asset = body["assets"][0]
    Path(asset["stored_path"]).unlink()
    response = client().get(
        f"/api/upload/{body['upload_id']}/assets/{asset['asset_id']}/preview",
        headers=bearer("token-alice"),
    )
    assert response.status_code == 410, response.text


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
