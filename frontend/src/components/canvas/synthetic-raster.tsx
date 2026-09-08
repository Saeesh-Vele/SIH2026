/**
 * Placeholder raster. Until /upload streams real GeoTIFF tiles, the canvas
 * paints deterministic fractal noise mapped into a dark, cold ramp — close
 * enough to a night-pass capture to make overlay geometry legible without
 * pretending to be a real scene. SAR gets a finer, higher-contrast ramp to read
 * as speckle; optical gets a coarser, faintly olive one to read as terrain.
 */
const RAMPS = {
  optical: {
    baseFrequency: "0.014 0.019",
    octaves: 6,
    r: "0.04 0.07 0.11 0.16 0.21",
    g: "0.05 0.09 0.14 0.20 0.26",
    b: "0.06 0.09 0.13 0.17 0.22",
  },
  sar: {
    baseFrequency: "0.07",
    octaves: 3,
    r: "0.03 0.06 0.12 0.22 0.38",
    g: "0.03 0.07 0.13 0.24 0.41",
    b: "0.04 0.08 0.15 0.26 0.43",
  },
} as const;

export function SyntheticRaster({
  seed,
  tint,
  className,
}: {
  seed: number;
  tint: "optical" | "sar";
  className?: string;
}) {
  const filterId = `raster-${tint}-${seed}`;
  const ramp = RAMPS[tint];

  return (
    <svg
      className={className}
      viewBox="0 0 400 400"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden
    >
      <defs>
        <filter
          id={filterId}
          x="0"
          y="0"
          width="100%"
          height="100%"
          colorInterpolationFilters="sRGB"
        >
          <feTurbulence
            type="fractalNoise"
            baseFrequency={ramp.baseFrequency}
            numOctaves={ramp.octaves}
            seed={seed}
            result="noise"
          />
          <feColorMatrix in="noise" type="saturate" values="0" result="grey" />
          <feComponentTransfer in="grey">
            <feFuncR type="table" tableValues={ramp.r} />
            <feFuncG type="table" tableValues={ramp.g} />
            <feFuncB type="table" tableValues={ramp.b} />
            <feFuncA type="discrete" tableValues="1" />
          </feComponentTransfer>
        </filter>
      </defs>
      <rect width="400" height="400" filter={`url(#${filterId})`} />
    </svg>
  );
}
