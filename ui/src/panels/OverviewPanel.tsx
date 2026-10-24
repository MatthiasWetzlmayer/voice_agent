import type { AgentState, TaskStatus } from "../api";

const counters: { label: string; statuses: TaskStatus[]; tone: string }[] = [
  { label: "Laufend", statuses: ["läuft"], tone: "running" },
  { label: "Warten auf Antwort", statuses: ["wartet auf Antwort"], tone: "waiting" },
  { label: "Fertig", statuses: ["fertig"], tone: "done" },
  { label: "Fehlgeschlagen / abgebrochen", statuses: ["fehlgeschlagen", "abgebrochen"], tone: "failed" },
];

export function OverviewPanel({ state }: { state: AgentState }) {
  return (
    <div className="stats">
      {counters.map(({ label, statuses, tone }) => (
        <div key={label} className={`stat tone-${tone}`}>
          <span className="stat-value">{state.tasks.filter((t) => statuses.includes(t.status)).length}</span>
          <span className="stat-label">{label}</span>
        </div>
      ))}
    </div>
  );
}
