"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";

/**
 * The globe, orbit and particles: never rendered on the server, and not even
 * requested until the browser is idle after load. The headline and buttons
 * are interactive long before the planet appears; on a slow phone that is the
 * difference between a page that responds and one that is busy drawing Earth.
 */
const HeroVisual = dynamic(() => import("./hero-visual"), { ssr: false, loading: () => null });

export function HeroVisualLazy() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const start = () => setReady(true);
    if ("requestIdleCallback" in window) {
      const id = window.requestIdleCallback(start, { timeout: 2500 });
      return () => window.cancelIdleCallback(id);
    }
    const id = setTimeout(start, 1200);
    return () => clearTimeout(id);
  }, []);

  return ready ? <HeroVisual /> : null;
}
