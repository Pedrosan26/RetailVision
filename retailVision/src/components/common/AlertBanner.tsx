// The crowding alert, shown above whatever page is open once a zone has
// held at or over the configured limit for the configured minutes. Rendered
// in the shell rather than a page so it cannot be scrolled away or missed
// by being on the wrong tab; status colour is paired with words, never
// alone.

import { useOccupancyAlert } from "../../hooks/useOccupancyAlert";
import { useUiStore } from "../../store/uiStore";

/** Renders a critical banner for every zone currently over the occupancy limit. */
export function AlertBanner() {
  const alerts = useOccupancyAlert();
  const alertLimit = useUiStore((state) => state.alertLimit);

  if (alerts.length === 0) return null;

  return (
    <div className="mb-5 flex flex-col gap-2">
      {alerts.map((alert) => (
        <div
          key={alert.zone_id}
          role="alert"
          className="flex flex-wrap items-baseline gap-x-2 rounded-md border border-[var(--app-critical)] bg-[var(--app-critical-wash)] px-4 py-2.5 text-sm"
        >
          <span className="font-semibold text-[var(--app-critical)]">Crowding:</span>
          <span className="text-[var(--app-ink)]">
            {alert.zone_id} has held {alert.total} people for {alert.minutes}m
          </span>
          <span className="text-[var(--app-ink-muted)]">(limit {alertLimit})</span>
        </div>
      ))}
    </div>
  );
}

/** Renders the two-field alert configuration, for the Overview page. */
export function AlertSettings() {
  const { alertLimit, alertMinutes, setAlertLimit, setAlertMinutes } = useUiStore();

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-[var(--app-ink-secondary)]">
      <label className="inline-flex items-center gap-2">
        Alert when a zone holds
        <input
          type="number"
          min={1}
          value={alertLimit ?? ""}
          placeholder="off"
          onChange={(e) => setAlertLimit(e.target.value === "" ? null : Math.max(1, Number(e.target.value)))}
          className="w-16 rounded-md border border-[var(--app-line-strong)] bg-[var(--app-raised)] px-2 py-1 text-sm tabular-nums text-[var(--app-ink)]"
        />
        or more people for
      </label>
      <label className="inline-flex items-center gap-2">
        <input
          type="number"
          min={1}
          value={alertMinutes}
          onChange={(e) => setAlertMinutes(Math.max(1, Number(e.target.value)))}
          className="w-14 rounded-md border border-[var(--app-line-strong)] bg-[var(--app-raised)] px-2 py-1 text-sm tabular-nums text-[var(--app-ink)]"
        />
        minutes
      </label>
      {alertLimit == null && <span className="text-xs text-[var(--app-ink-muted)]">alerts are off</span>}
    </div>
  );
}
