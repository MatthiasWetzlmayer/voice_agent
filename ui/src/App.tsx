import { useAgentState } from "./api";
import { panels } from "./panels";

export function App() {
  const { state, connected } = useAgentState();

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>◉</span>
          <div>
            <h1>Voice Assistant</h1>
            {state && (
              <p className="models">
                Voice: {state.agent.voice_model} · Tasks: {state.agent.task_model}
              </p>
            )}
          </div>
        </div>
        <span className={`connection ${connected ? "online" : "offline"}`}>
          {connected ? "Agent verbunden" : "Agent nicht erreichbar"}
        </span>
      </header>

      <main className="content">
        {!state ? (
          <p className="empty">
            Warte auf den Agenten … Er muss lokal laufen: <code>cd agent && uv run voice-assistant</code>
          </p>
        ) : (
          panels.map(({ id, title, component: Panel }) => (
            <section key={id} className="panel">
              <h2>{title}</h2>
              <Panel state={state} />
            </section>
          ))
        )}
      </main>
    </div>
  );
}
