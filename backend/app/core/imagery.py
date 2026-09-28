"""What the console needs to show an uploaded file: its georeferencing, and a
picture of it.

Both read the file the way the models do. The preview is the same
percentile-stretched true-colour rendering fusion hands the VQA model
(`fusion.rgb_preview`), so what a person sees is what the model saw.
"""

from __future__ import annotations

import io
import logging
import warnings
from pathlib import Path
from typing import Any

from app.models.schemas import AssetGeo

logger = logging.getLogger(__name__)

#: Formats that cannot carry georeferencing at all. For these the console says
#: "No georeferencing in this file"; for anything else it could not read, it
#: says "Not read" rather than guessing.
PLAIN_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

#: Longest side of a rendered preview. Large GeoTIFFs are downsampled; small
#: benchmark tiles (EuroSAT is 64 px) are kept at native size and scaled up in
#: the browser without smoothing.
PREVIEW_MAX_SIDE = 1024


def raster_metadata(path: str | Path) -> AssetGeo:
    """Size, bands and — when the header has them — CRS, centre and pixel size."""
    path = Path(path)
    plain = path.suffix.lower() in PLAIN_IMAGE_SUFFIXES

    try:
        import rasterio
        import rasterio.errors
        from rasterio.warp import transform_bounds
    except ImportError:  # pragma: no cover - rasterio is in requirements
        return AssetGeo(status="none" if plain else "not_read")

    try:
        with warnings.catch_warnings():
            # A PNG having no geotransform is the expected case, not news.
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            src = rasterio.open(path)
        with src:
            width, height, bands = int(src.width), int(src.height), int(src.count)
            if plain or src.crs is None:
                return AssetGeo(status="none", width=width, height=height, bands=bands)

            epsg = src.crs.to_epsg()
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            gsd = None
            if src.crs.is_projected and src.crs.linear_units in {"metre", "meter", "m"}:
                gsd = round(float(abs(src.res[0])), 3)
            return AssetGeo(
                status="georeferenced",
                width=width,
                height=height,
                bands=bands,
                crs=f"EPSG:{epsg}" if epsg else src.crs.to_string(),
                lat=round((south + north) / 2, 6),
                lon=round((west + east) / 2, 6),
                gsd_m=gsd,
            )
    except Exception as exc:  # noqa: BLE001 - an unreadable header is reported, not raised
        logger.info("could not read raster metadata from %s: %s", path.name, exc)
        return AssetGeo(status="none" if plain else "not_read")


def preview_path(stored_path: str | Path) -> Path:
    stored = Path(stored_path)
    return stored.with_name(f"{stored.stem}.preview.png")


def render_preview(stored_path: str | Path) -> bytes:
    """A PNG of the asset as the models see it, cached beside the upload."""
    from app.models.fusion import _read_bands, rgb_preview

    cached = preview_path(stored_path)
    if cached.exists():
        return cached.read_bytes()

    image, bands = _read_bands(stored_path)
    picture: Any = rgb_preview(image, bands)
    if max(picture.size) > PREVIEW_MAX_SIDE:
        picture.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))

    buffer = io.BytesIO()
    picture.save(buffer, format="PNG", optimize=True)
    data = buffer.getvalue()
    try:
        cached.write_bytes(data)
    except OSError:  # a read-only store still gets its preview, just uncached
        logger.info("could not cache preview for %s", Path(stored_path).name)
    return data
