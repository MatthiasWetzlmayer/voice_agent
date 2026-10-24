import type { AgentState, Task } from "../api";
import { TaskCard } from "../components/TaskCard";

const active = (task: Task) => task.status === "läuft" || task.status === "wartet auf Antwort";

export function TasksPanel({ state }: { state: AgentState }) {
  if (state.tasks.length === 0) {
    return (
      <p className="empty">
        Noch keine Tasks. Bitte den Assistenten um eine längere Aufgabe, z.B. „Recherchier mir etwas zu Wärmepumpen“.
      </p>
    );
  }
  // Aktive Tasks zuerst, innerhalb davon die neuesten oben.
  const tasks = [...state.tasks].sort((a, b) => Number(active(b)) - Number(active(a)) || b.started - a.started);
  return (
    <div className="task-grid">
      {tasks.map((task) => (
        <TaskCard key={task.id} task={task} />
      ))}
    </div>
  );
}
