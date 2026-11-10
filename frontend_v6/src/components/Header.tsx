export type BackendState = "unconfigured" | "checking" | "connected" | "unreachable";

const LABEL: Record<BackendState, string> = {
  unconfigured: "Backend not configured",
  checking: "Checking backend",
  connected: "Backend connected",
  unreachable: "Backend unreachable",
};

export function HexMark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2.2 20.5 7.1v9.8L12 21.8 3.5 16.9V7.1Z" fill="none" stroke="var(--primary)" strokeWidth="1.4" strokeLinejoin="round" />
      <path d="M12 7.2 16.3 9.7v4.6L12 16.8 7.7 14.3V9.7Z" fill="var(--primary)" fillOpacity="0.22" stroke="var(--primary)" strokeWidth="1" strokeLinejoin="round" />
      <circle cx="12" cy="12" r="1.4" fill="var(--primary)" />
    </svg>
  );
}

export default function Header({ backend, apiUrl, onRecheck, runs }: { runs: number; backend: BackendState; apiUrl: string | null; onRecheck: () => void }) {
  const dot =
    backend === "connected" ? "bg-success" : backend === "checking" ? "bg-warning anim-pulse" : backend === "unreachable" ? "bg-destructive" : "bg-muted-foreground";
  return (
    <header className="sticky top-0 z-20 border-b border-border bg-background/85 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1440px] items-center justify-between gap-4 px-4 md:px-8">
        <div className="flex items-center gap-3">
          <HexMark />
          <span className="text-[15px] font-semibold tracking-tight">DynaCPI</span>
          <span className="hidden h-4 w-px bg-border sm:block" />
          <span className="micro hidden sm:block">Sequence-based CPI</span>
        </div>
        <button
          type="button"
          onClick={onRecheck}
          title={apiUrl ? `${apiUrl} — click to re-check` : "Set VITE_DYNACPI_API_URL to connect"}
          className="group flex items-center gap-2.5 rounded-md border border-border bg-card px-3 py-1.5 transition-colors hover:border-primary/40"
        >
          <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${dot}`} />
          <span className="micro !text-foreground/90" role="status" aria-live="polite">
            {LABEL[backend]}
          </span>
          <span className="micro hidden border-l border-border pl-2.5 md:block">Runs this session: {runs}</span>
        </button>
      </div>
    </header>
  );
}