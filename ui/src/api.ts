import { useEffect, useState } from "react";

// Spiegelt die Antwort von GET /api/state des Agenten (agent/src/voice_assistant/dashboard.py).

/** Adresse des lokal laufenden Agenten, änderbar über VITE_AGENT_URL in ui/.env. */
export const AGENT_URL = import.meta.env.VITE_AGENT_URL ?? "http://127.0.0.1:8765";

export type TaskStatus = "läuft" | "wartet auf Antwort" | "fertig" | "fehlgeschlagen" | "abgebrochen";

export interface Todo {
  content: string;
  status: "pending" | "in_progress" | "completed";
}

export interface Step {
  tool: string;
  args: Record<string, unknown>;
  time: number; // Unix-Zeit in Sekunden
}

export interface Task {
  id: string;
  name: string;
  description: string;
  status: TaskStatus;
  started: number;
  finished: number | null;
  question: string | null;
  result: string | null;
  error: string | null;
  todos: Todo[];
  steps: Step[];
}

export interface AgentState {
  agent: { voice_model: string; task_model: string; started: number };
  tasks: Task[];
}

export interface Connection {
  state: AgentState | null; // letzter bekannter Zustand, bleibt bei Verbindungsabbruch stehen
  connected: boolean;
}

/** Fragt den Agenten regelmäßig ab. Die UI liest nur, sie verändert nichts am Agenten. */
export function useAgentState(intervalMs = 1000): Connection {
  const [connection, setConnection] = useState<Connection>({ state: null, connected: false });

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const response = await fetch(`${AGENT_URL}/api/state`);
        if (!response.ok) throw new Error(String(response.status));
        const state = (await response.json()) as AgentState;
        if (active) setConnection({ state, connected: true });
      } catch {
        if (active) setConnection((previous) => ({ ...previous, connected: false }));
      }
    };
    poll();
    const timer = setInterval(poll, intervalMs);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [intervalMs]);

  return connection;
}

/** Aktuelle Zeit in Sekunden, aktualisiert sich jede Sekunde (für Laufzeiten). */
export function useNow(): number {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  }, []);
  return now;
}

export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}
