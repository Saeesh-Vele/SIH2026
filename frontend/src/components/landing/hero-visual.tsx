"use client";

import type { COBEOptions } from "cobe";
import { useReducedMotion } from "motion/react";
import { Globe } from "@/components/ui/globe";
import { Particles } from "@/components/ui/particles";

/**
 * Night-side Earth in the deck's blue, with a satellite on a tilted orbit.
 *
 * cobe draws the planet on one WebGL canvas (a few KB); react-three-fiber
 * would ship three.js — well over 150 KB gzipped — for the same sphere. The
 * orbit is plain CSS: a ring behind the globe, the near half of the same ring
 * in front of it, and the satellite riding `offset-path` around both.
 */
const GLOBE: COBEOptions = {
  width: 1200,
  height: 1200,
  onRender: () => {},
  devicePixelRatio: 2,
  phi: 1.2,
  theta: 0.28,
  dark: 1,
  diffuse: 1.15,
  mapSamples: 16000,
  mapBrightness: 5.5,
  baseColor: [0.11, 0.16, 0.23],
  markerColor: [34 / 255, 201 / 255, 192 / 255],
  glowColor: [0.16, 0.3, 0.55],
  markers: [],
};

export default function HeroVisual() {
  const reduceMotion = useReducedMotion();

  return (
    <div className="pointer-events-none absolute inset-0" aria-hidden>
      {reduceMotion ? null : (
        <Particles
          className="absolute inset-0"
          quantity={70}
          staticity={80}
          ease={90}
          size={0.35}
          color="#c6d4df"
          vx={0.02}
          vy={-0.01}
        />
      )}

      <div className="absolute left-1/2 top-[-6%] aspect-square w-[min(115vw,560px)] -translate-x-1/2 opacity-60 md:left-auto md:right-[-12%] md:top-1/2 md:w-[min(62vw,780px)] md:translate-x-0 md:-translate-y-1/2 md:opacity-100">
        {/* Far half of the orbit, drawn under the planet. */}
        <Orbit />
        <div className="pointer-events-auto absolute inset-0">
          <Globe config={GLOBE} speed={0.0018} className="max-w-none" />
        </div>
        {/* Near half of the same orbit, over the planet. */}
        <Orbit near />
      </div>
    </div>
  );
}

function Orbit({ near = false }: { near?: boolean }) {
  return (
    <div className="absolute inset-[-8%] rotate-[-17deg]">
      <div
        className="absolute inset-x-0 top-1/2 h-[30%] -translate-y-1/2 rounded-[50%] border border-uplink/35"
        style={near ? { clipPath: "inset(50% 0 0 0)" } : undefined}
      >
        {near ? <Satellite /> : null}
      </div>
    </div>
  );
}

/**
 * Rides the ellipse. It lives inside the near ring, which is clipped to its
 * lower half, so on the far side of the orbit it is simply out of sight.
 */
function Satellite() {
  return (
    <span
      className="animate-orbit absolute left-0 top-0 block"
      style={{
        offsetPath: "ellipse(50% 50% at 50% 50%)",
        offsetRotate: "0deg",
        offsetDistance: "30%",
      }}
    >
      <svg viewBox="0 0 28 14" className="h-3.5 w-7 drop-shadow-[0_0_6px_rgb(34_201_192/0.8)]">
        <rect x="0" y="4" width="9" height="6" fill="#4a8cff" />
        <rect x="19" y="4" width="9" height="6" fill="#4a8cff" />
        <rect x="10" y="2.5" width="8" height="9" rx="1" fill="#22c9c0" />
        <rect x="9" y="6.5" width="10" height="1" fill="#c6d4df" />
      </svg>
    </span>
  );
}
