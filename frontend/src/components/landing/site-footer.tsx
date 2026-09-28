import { Wordmark } from "@/components/landing/wordmark";

const REPO_URL = "https://github.com/Saeesh-Vele/SIH2026";

export function SiteFooter() {
  return (
    <footer className="border-t border-rule">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-10 text-[13.5px] text-muted-foreground sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="space-y-2">
          <Wordmark />
          <p>
            Built by Synac Syndicate for Smart India Hackathon 2026, problem statement 26167.
          </p>
        </div>
        <a
          href={REPO_URL}
          className="self-start rounded-sm text-text-dim underline-offset-4 transition-colors hover:text-signal hover:underline sm:self-auto"
        >
          Source on GitHub
        </a>
      </div>
    </footer>
  );
}
