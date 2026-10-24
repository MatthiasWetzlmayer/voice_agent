import type { FC } from "react";

import type { AgentState } from "../api";
import { OverviewPanel } from "./OverviewPanel";
import { TasksPanel } from "./TasksPanel";

export interface Panel {
  id: string;
  title: string;
  component: FC<{ state: AgentState }>;
}

/**
 * Alle Ansichten des Dashboards, in Anzeigereihenfolge.
 * Neue Ansicht: Komponente in diesem Ordner anlegen und hier eintragen.
 * Braucht sie neue Daten, liefert der Agent sie zusätzlich in snapshot() (assistant.py)
 * und der Typ AgentState in api.ts bekommt das passende Feld.
 */
export const panels: Panel[] = [
  { id: "overview", title: "Übersicht", component: OverviewPanel },
  { id: "tasks", title: "Hintergrund-Tasks", component: TasksPanel },
];
