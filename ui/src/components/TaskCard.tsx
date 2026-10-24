import { formatDuration, useNow, type Task } from "../api";
import { StatusBadge } from "./StatusBadge";

function describeArgs(args: Record<string, unknown>): string {
  return Object.values(args)
    .map((value) => {
      if (typeof value === "string") return value;
      if (Array.isArray(value)) return `${value.length} Einträge`;
      return JSON.stringify(value);
    })
    .join(", ");
}

export function TaskCard({ task }: { task: Task }) {
  const now = useNow();
  const duration = (task.finished ?? now) - task.started;
  const recentSteps = task.steps.slice(-5).reverse();

  return (
    <article className="task-card">
      <header className="task-head">
        <h3>{task.name}</h3>
        <StatusBadge status={task.status} />
      </header>
      <p className="task-meta">
        {task.finished ? "Dauer" : "Läuft seit"} {formatDuration(duration)} · {task.steps.length} Schritte
      </p>
      <p className="task-description">{task.description}</p>

      {task.question && (
        <div className="callout tone-waiting">
          <strong>Wartet auf Freigabe</strong> {task.question}
        </div>
      )}
      {task.result && <div className="callout tone-done">{task.result}</div>}
      {task.error && <div className="callout tone-failed">{task.error}</div>}

      {task.todos.length > 0 && (
        <section>
          <h4>Plan</h4>
          <ul className="todos">
            {task.todos.map((todo, i) => (
              <li key={i} className={`todo todo-${todo.status}`}>
                <span className="todo-mark" aria-hidden /> {todo.content}
              </li>
            ))}
          </ul>
        </section>
      )}

      {recentSteps.length > 0 && (
        <section>
          <h4>Letzte Schritte</h4>
          <ol className="steps">
            {recentSteps.map((step, i) => (
              <li key={i}>
                <code>{step.tool}</code>
                <span className="step-args">{describeArgs(step.args)}</span>
                <time>{new Date(step.time * 1000).toLocaleTimeString("de-DE")}</time>
              </li>
            ))}
          </ol>
        </section>
      )}
    </article>
  );
}
