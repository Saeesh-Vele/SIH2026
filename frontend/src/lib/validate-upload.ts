/**
 * Client-side file gate. GeoTIFF is the working format; PNG/JPEG are let
 * through only in benchmark dataset mode, where the corpora ship as RGB.
 * Mirrors the same rule in backend/app/routes/upload.py.
 */

export const GEOTIFF_EXTENSIONS = [".tif", ".tiff", ".gtiff"] as const;
export const BENCHMARK_EXTENSIONS = [".png", ".jpg", ".jpeg"] as const;

export const MAX_UPLOAD_BYTES = 512 * 1024 * 1024;

export type ValidationResult = { ok: true } | { ok: false; reason: string };

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot).toLowerCase();
}

export function acceptAttribute(benchmarkMode: boolean): string {
  const exts = benchmarkMode
    ? [...GEOTIFF_EXTENSIONS, ...BENCHMARK_EXTENSIONS]
    : [...GEOTIFF_EXTENSIONS];
  return [...exts, "image/tiff", benchmarkMode ? "image/png,image/jpeg" : ""]
    .filter(Boolean)
    .join(",");
}

export function validateFile(file: File, benchmarkMode: boolean): ValidationResult {
  const ext = extensionOf(file.name);

  if (!ext) {
    return { ok: false, reason: "No file extension. Expected a GeoTIFF (.tif)." };
  }

  if ((GEOTIFF_EXTENSIONS as readonly string[]).includes(ext)) {
    // fall through to the size check
  } else if ((BENCHMARK_EXTENSIONS as readonly string[]).includes(ext)) {
    if (!benchmarkMode) {
      return {
        ok: false,
        reason: `${ext} is only accepted in benchmark dataset mode. Turn it on, or supply a GeoTIFF.`,
      };
    }
  } else {
    return { ok: false, reason: `${ext} is not a supported format. Expected .tif or .tiff.` };
  }

  if (file.size === 0) {
    return { ok: false, reason: "File is empty." };
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    return {
      ok: false,
      reason: `${formatBytes(file.size)} exceeds the ${formatBytes(MAX_UPLOAD_BYTES)} limit.`,
    };
  }

  return { ok: true };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`;
}
