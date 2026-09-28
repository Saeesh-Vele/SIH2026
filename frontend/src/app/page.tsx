import type { Metadata } from "next";
import { AskExample } from "@/components/landing/ask-example";
import { Capabilities } from "@/components/landing/capabilities";
import { Hero } from "@/components/landing/hero";
import { Pipeline } from "@/components/landing/pipeline";
import { SiteFooter } from "@/components/landing/site-footer";
import { SiteHeader } from "@/components/landing/site-header";
import { UnderTheHood } from "@/components/landing/under-the-hood";

export const metadata: Metadata = {
  title: "SatQuery AI — ask satellite imagery a question",
};

export default function LandingPage() {
  return (
    <div className="texture-starfield min-h-dvh bg-void">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-sm focus:bg-panel focus:px-3 focus:py-2 focus:text-foreground"
      >
        Skip to content
      </a>
      <SiteHeader />
      <main id="main" className="-mt-16">
        <Hero />
        <AskExample />
        <Capabilities />
        <Pipeline />
        <UnderTheHood />
      </main>
      <SiteFooter />
    </div>
  );
}
