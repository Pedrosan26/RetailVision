// Sections a reader can reorder to taste, with move controls that appear on
// hover at each section's top-right corner. Deliberately up/down buttons
// rather than drag-and-drop: reordering page sections is a set-and-forget
// act, not a frequent gesture, so it does not justify a drag library -- and
// buttons work for keyboard users, which a drag handle does not.
//
// The chosen order persists per-browser alongside the alert settings: how a
// page is arranged is a property of the person reading it, not of the
// deployment. Sections added by later versions simply append in their
// default position rather than invalidating a stored order.

import type { ReactNode } from "react";
import { useUiStore } from "../../store/uiStore";

export interface ArrangeableSection {
  /** Stable identifier the stored order refers to; also the accessible name of the controls. */
  id: string;
  children: ReactNode;
}

/** Renders sections in the reader's stored order, each with hover controls to move it up or down. */
export function ArrangeableSections({ page, sections }: { page: string; sections: ArrangeableSection[] }) {
  const stored = useUiStore((state) => state.sectionOrder[page]);
  const setSectionOrder = useUiStore((state) => state.setSectionOrder);

  const ids = sections.map((section) => section.id);
  const order = [
    ...(stored ?? []).filter((id) => ids.includes(id)),
    ...ids.filter((id) => !(stored ?? []).includes(id)),
  ];

  /** Swaps a section with its neighbour in the given direction. */
  function move(id: string, delta: number) {
    const from = order.indexOf(id);
    const to = from + delta;
    if (to < 0 || to >= order.length) return;
    const next = [...order];
    [next[from], next[to]] = [next[to], next[from]];
    setSectionOrder(page, next);
  }

  const buttonClass =
    "rounded border border-[var(--app-line-strong)] bg-[var(--app-raised)] px-1.5 py-0.5 text-xs " +
    "text-[var(--app-ink-muted)] hover:bg-[var(--app-accent-wash)] hover:text-[var(--app-accent)] disabled:opacity-30";

  return (
    <>
      {order.map((id, index) => {
        const section = sections.find((candidate) => candidate.id === id);
        if (!section) return null;
        return (
          <section key={id} className="group relative">
            <div className="absolute -top-1 right-0 z-10 flex gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
              <button
                type="button"
                aria-label={`Move ${id} up`}
                title={`Move ${id} up`}
                disabled={index === 0}
                onClick={() => move(id, -1)}
                className={buttonClass}
              >
                ↑
              </button>
              <button
                type="button"
                aria-label={`Move ${id} down`}
                title={`Move ${id} down`}
                disabled={index === order.length - 1}
                onClick={() => move(id, 1)}
                className={buttonClass}
              >
                ↓
              </button>
            </div>
            {section.children}
          </section>
        );
      })}
    </>
  );
}
