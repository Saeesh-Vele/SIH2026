/** Degrees-minutes-seconds, the notation analysts read off a chart. */
export function formatDms(value: number, axis: "lat" | "lon"): string {
  const hemisphere =
    axis === "lat" ? (value >= 0 ? "N" : "S") : value >= 0 ? "E" : "W";
  const abs = Math.abs(value);
  const deg = Math.floor(abs);
  const minFloat = (abs - deg) * 60;
  const min = Math.floor(minFloat);
  const sec = ((minFloat - min) * 60).toFixed(1);
  return `${hemisphere} ${deg}°${String(min).padStart(2, "0")}'${sec.padStart(4, "0")}"`;
}

export function formatUtc(iso: string): string {
  return iso.replace("T", " ").replace("Z", "Z");
}

export function formatDuration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms}ms`;
}
