import type { TaskStatus } from "@/lib/api";

const tones: Record<TaskStatus, string> = {
  läuft: "running",
  "wartet auf Antwort": "waiting",
  fertig: "done",
  fehlgeschlagen: "failed",
  abgebrochen: "cancelled",
};

export function StatusBadge({ status }: { status: TaskStatus }) {
  return <span className={`badge tone-${tones[status]}`}>{status}</span>;
}
