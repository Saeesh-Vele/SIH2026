/**
 * What actually runs, stated plainly. Every row is checked against the repo:
 * model_config.yaml, the adapter in backend/checkpoints, and the README.
 */
const SPEC: { term: string; detail: React.ReactNode }[] = [
  {
    term: "Questions and captions",
    detail: (
      <>
        LLaVA-1.5 (7B), LoRA fine-tuned on EuroSAT: Sentinel-2 land-cover
        scenes.
      </>
    ),
  },
  {
    term: "Change detection",
    detail: (
      <>
        LLaVA&rsquo;s own vision encoder compares the two dates patch by patch to find what
        changed, then LLaVA describes it.
      </>
    ),
  },
  {
    term: "Optical + SAR fusion",
    detail: (
      <>
        Two SSL4EO-S12 encoders read Sentinel-2 (13 bands) and Sentinel-1 radar (VV and VH).
        Their combined reading goes to LLaVA with your question.
      </>
    ),
  },
  {
    term: "Understanding the question",
    detail: (
      <>
        A language model classifies what you&rsquo;re asking. Without one configured, keyword
        rules do it, and the trace says so.
      </>
    ),
  },
  {
    term: "Imagery",
    detail: (
      <>
        GeoTIFF, keeping every band. PNG and JPEG in benchmark mode, for datasets that ship as
        plain images.
      </>
    ),
  },
  {
    term: "Hardware",
    detail: (
      <>
        LLaVA runs in 4-bit on a CUDA GPU. Where there is none, a smaller CPU model can answer
        instead, and every such answer is labelled Degraded.
      </>
    ),
  },
  {
    term: "Not yet",
    detail: (
      <>
        It doesn&rsquo;t point to individual objects in an image. Changed areas are marked as
        rectangles around the patches that changed, not traced outlines.
      </>
    ),
  },
];

export function UnderTheHood() {
  return (
    <section className="mx-auto w-full max-w-6xl px-4 py-20 sm:px-6 md:py-28">
      <div className="grid gap-10 md:grid-cols-[minmax(0,4fr)_minmax(0,8fr)] md:gap-16">
        <div>
          <h2 className="text-balance text-[clamp(1.75rem,3.4vw,2.5rem)] font-semibold leading-[1.1] tracking-[-0.025em] text-foreground">
            Under the hood
          </h2>
          <p className="mt-5 max-w-sm text-[16px] leading-relaxed text-text-dim">
            The models and data behind each answer, including what it can&rsquo;t do yet.
          </p>
        </div>

        <dl className="divide-y divide-rule border-y border-rule">
          {SPEC.map((row) => (
            <div key={row.term} className="grid gap-1 py-4 sm:grid-cols-[13rem_1fr] sm:gap-6">
              <dt className="text-[14px] font-medium text-foreground">{row.term}</dt>
              <dd className="max-w-[60ch] text-[14.5px] leading-relaxed text-text-dim">
                {row.detail}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
