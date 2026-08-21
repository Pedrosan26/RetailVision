// Shared interface primitives.
//
// Every panel in the dashboard is one of a small number of shapes, so they
// are defined once here rather than as repeated Tailwind strings. That keeps
// spacing and hairlines consistent, and means a change to the panel treatment
// is one edit rather than a search.
//
// Colours come from the token variables in index.css via arbitrary values, so
// a single class covers light and dark rather than every element carrying a
// dark: twin.

import type { ReactNode } from "react";

/** A bordered panel, the default container for anything on a page. */
export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-md border border-[var(--app-line)] bg-[var(--app-surface)] ${className}`}
    >
      {children}
    </div>
  );
}

/** A card's heading row: a title, optional description, and optional controls on the right. */
export function CardHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[var(--app-line)] px-4 py-3">
      <div className="min-w-0">
        <h3 className="text-sm font-semibold text-[var(--app-ink)]">{title}</h3>
        {description && <p className="mt-0.5 text-xs text-[var(--app-ink-muted)]">{description}</p>}
      </div>
      {actions}
    </div>
  );
}

/** A page's title block. */
export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-[var(--app-ink)]">{title}</h1>
        {description && (
          <p className="mt-1 max-w-2xl text-sm text-[var(--app-ink-secondary)]">{description}</p>
        )}
      </div>
      {actions}
    </div>
  );
}

/** A section heading between cards, quieter than a page title. */
export function SectionHeading({ children, hint }: { children: ReactNode; hint?: string }) {
  return (
    <div className="mb-3">
      <h2 className="text-[0.7rem] font-semibold uppercase tracking-[0.08em] text-[var(--app-ink-muted)]">
        {children}
      </h2>
      {hint && <p className="mt-1 text-xs text-[var(--app-ink-muted)]">{hint}</p>}
    </div>
  );
}

type Tone = "neutral" | "good" | "warning" | "critical" | "accent";

const TONE_STYLE: Record<Tone, string> = {
  neutral: "border-[var(--app-line-strong)] text-[var(--app-ink-secondary)]",
  good: "border-[var(--app-good)] text-[var(--app-good)]",
  warning: "border-[var(--app-warning)] text-[var(--app-ink-secondary)]",
  critical: "border-[var(--app-critical)] text-[var(--app-critical)]",
  accent: "border-[var(--app-accent)] text-[var(--app-accent)]",
};

/** A small labelled tag. Status tones always carry their label, so meaning never rests on colour. */
export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: Tone }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[0.7rem] font-medium ${TONE_STYLE[tone]}`}
    >
      {children}
    </span>
  );
}

/** A headline figure with its label and optional supporting detail. */
export function StatTile({
  label,
  value,
  unit,
  detail,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  detail?: ReactNode;
}) {
  return (
    <div>
      <div className="text-[0.7rem] font-medium uppercase tracking-[0.08em] text-[var(--app-ink-muted)]">
        {label}
      </div>
      <div className="mt-1 flex items-baseline gap-1.5">
        <span className="text-2xl font-semibold tabular-nums tracking-tight text-[var(--app-ink)]">{value}</span>
        {unit && <span className="text-sm text-[var(--app-ink-muted)]">{unit}</span>}
      </div>
      {detail && <div className="mt-1 text-xs text-[var(--app-ink-muted)]">{detail}</div>}
    </div>
  );
}

/** Shown where content would be, when there is none. Says why, not just that. */
export function EmptyState({ title, hint }: { title: string; hint?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-[var(--app-line-strong)] px-6 py-10 text-center">
      <p className="text-sm text-[var(--app-ink-secondary)]">{title}</p>
      {hint && <p className="mx-auto mt-1.5 max-w-md text-xs text-[var(--app-ink-muted)]">{hint}</p>}
    </div>
  );
}

/** A labelled form control, for the filter rows above tables. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[0.7rem] font-medium uppercase tracking-[0.08em] text-[var(--app-ink-muted)]">
        {label}
      </span>
      {children}
    </label>
  );
}

export const CONTROL_CLASS =
  "rounded-md border border-[var(--app-line-strong)] bg-[var(--app-raised)] px-2.5 py-1.5 text-sm text-[var(--app-ink)]";

/** A row of mutually exclusive options, used for time ranges. */
export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: Array<{ value: T; label: string }>;
  value: T;
  onChange: (value: T) => void;
  ariaLabel: string;
}) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className="inline-flex rounded-md border border-[var(--app-line-strong)] p-0.5"
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option.value)}
            className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
              active
                ? "bg-[var(--app-accent)] text-[var(--app-accent-ink)]"
                : "text-[var(--app-ink-secondary)] hover:bg-[var(--app-accent-wash)]"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/** A live-ness dot, paired with text so it is never colour-alone. */
export function LiveDot({ live }: { live: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-[var(--app-ink-muted)]">
      <span
        aria-hidden
        className="inline-block h-1.5 w-1.5 rounded-full"
        style={{ background: live ? "var(--app-good)" : "var(--app-ink-muted)" }}
      />
      {live ? "live" : "idle"}
    </span>
  );
}
