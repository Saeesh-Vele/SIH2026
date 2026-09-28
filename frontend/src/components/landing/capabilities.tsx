"use client";

import { useState } from "react";
import { useReducedMotion } from "motion/react";
import { BlurFade } from "@/components/ui/blur-fade";
import { BorderBeam } from "@/components/ui/border-beam";
import { cn } from "@/lib/utils";

interface Capability {
  title: string;
  explain: string;
  ask: string;
  inputs: React.ReactNode;
  span: string;
}

/**
 * The four things the console does, each with a sketch of the imagery it
 * needs. The sketches match the upload slots in the console, so a visitor has
 * already seen the shape of "before + after" or "optical + SAR" by the time
 * they are asked to fill one.
 */
const CAPABILITIES: Capability[] = [
  {
    title: "Visual Q&A",
    explain: "Ask a direct question about one image and get a direct answer.",
    ask: "Is this area mostly farmland or forest?",
    inputs: <Frames frames={[{ label: "Image" }]} />,
    span: "md:col-span-3",
  },
  {
    title: "Scene captioning",
    explain: "Get a plain-English description of what one image shows.",
    ask: "Describe this scene.",
    inputs: <Frames frames={[{ label: "Image", lines: true }]} />,
    span: "md:col-span-2",
  },
  {
    title: "Change detection",
    explain:
      "Compare two images of the same place from different dates and see which areas changed.",
    ask: "What changed between these two dates?",
    inputs: <Frames frames={[{ label: "Before" }, { label: "After", marked: true }]} />,
    span: "md:col-span-2",
  },
  {
    title: "Optical + SAR fusion",
    explain:
      "Read an ordinary satellite image together with a radar (SAR) image of the same place. Radar sees through cloud and works at night.",
    ask: "What does the radar show that the photo doesn’t?",
    inputs: <Frames frames={[{ label: "Optical" }, { label: "SAR", speckle: true }]} />,
    span: "md:col-span-3",
  },
];

export function Capabilities() {
  return (
    <section className="mx-auto w-full max-w-6xl px-4 py-20 sm:px-6 md:py-28">
      <h2 className="max-w-xl text-balance text-[clamp(1.75rem,3.4vw,2.5rem)] font-semibold leading-[1.1] tracking-[-0.025em] text-foreground">
        Four kinds of question, one console
      </h2>
      <p className="mt-5 max-w-xl text-[16px] leading-relaxed text-text-dim">
        Pick a task yourself, or let SatQuery decide from the way you phrase the question.
      </p>

      <BlurFade inView className="mt-12 grid gap-3 md:grid-cols-5">
        {CAPABILITIES.map((capability) => (
          <Card key={capability.title} capability={capability} />
        ))}
      </BlurFade>
    </section>
  );
}

function Card({ capability }: { capability: Capability }) {
  const [active, setActive] = useState(false);
  const reduceMotion = useReducedMotion();

  return (
    <article
      onPointerEnter={() => setActive(true)}
      onPointerLeave={() => setActive(false)}
      className={cn(
        "relative flex flex-col gap-6 overflow-hidden rounded-(--radius-surface) border border-rule bg-panel p-6 transition-colors duration-(--dur-base) hover:border-rule-strong sm:p-7",
        capability.span,
      )}
    >
      {active && !reduceMotion ? (
        <BorderBeam size={90} duration={7} colorFrom="#4a8cff" colorTo="#22c9c0" />
      ) : null}

      <div className="h-24">{capability.inputs}</div>

      <div className="space-y-2">
        <h3 className="text-[18px] font-semibold tracking-[-0.01em] text-foreground">
          {capability.title}
        </h3>
        <p className="max-w-[46ch] text-[14.5px] leading-relaxed text-text-dim">
          {capability.explain}
        </p>
      </div>

      <p className="mt-auto flex gap-2 border-t border-rule pt-4 font-mono text-[12.5px] text-text-dim">
        <span className="text-signal" aria-hidden>
          ›
        </span>
        <span>
          <span className="sr-only">Example question: </span>
          {capability.ask}
        </span>
      </p>
    </article>
  );
}

interface FrameSpec {
  label: string;
  /** Caption lines under the frame: the captioning task's output. */
  lines?: boolean;
  /** A marked region: where change detection puts its mask. */
  marked?: boolean;
  /** Fine grain, the way SAR speckle reads. */
  speckle?: boolean;
}

/** Schematic image slots. Decorative; the text around them carries the meaning. */
function Frames({ frames }: { frames: FrameSpec[] }) {
  return (
    <div className="flex h-full items-end gap-3" aria-hidden>
      {frames.map((frame) => (
        <div key={frame.label} className="flex flex-col gap-1.5">
          <div
            className={cn(
              "texture-graticule relative size-[4.5rem] rounded-sm border border-rule-strong bg-raised",
              frame.speckle && "texture-scanline",
            )}
          >
            {frame.marked ? (
              <span className="absolute left-[38%] top-[30%] h-[34%] w-[40%] border border-signal bg-signal/15" />
            ) : null}
            {frame.lines ? (
              <span className="absolute inset-x-2 bottom-2 space-y-1">
                <span className="block h-0.5 w-4/5 bg-muted-foreground/50" />
                <span className="block h-0.5 w-3/5 bg-muted-foreground/50" />
              </span>
            ) : null}
          </div>
          <span className="font-mono text-[10.5px] text-muted-foreground">{frame.label}</span>
        </div>
      ))}
    </div>
  );
}
