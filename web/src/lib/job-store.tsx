"use client";

// Holds the most recent generated job so the Studio and Simulator pages share it.
// Kept in sessionStorage so a refresh does not lose it; storage can be
// unavailable (private mode), in which case it simply lives in memory.

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { GcodeStats } from "./api";

export type JobPage = { gcode: string; stats: GcodeStats };
export type CurrentJob = { id: string; title: string; model: string; writer: string | null; pages: JobPage[]; createdAt: number };

const KEY = "scriptor.job";
const Ctx = createContext<{ job: CurrentJob | null; setJob: (j: CurrentJob | null) => void }>({ job: null, setJob: () => {} });

export function JobProvider({ children }: { children: React.ReactNode }) {
  const [job, setJobState] = useState<CurrentJob | null>(null);
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(KEY);
      if (raw) setJobState(JSON.parse(raw));
    } catch {
      /* storage unavailable */
    }
  }, []);
  const setJob = useCallback((j: CurrentJob | null) => {
    setJobState(j);
    try {
      if (j) sessionStorage.setItem(KEY, JSON.stringify(j));
      else sessionStorage.removeItem(KEY);
    } catch {
      /* storage full or unavailable */
    }
  }, []);
  return <Ctx.Provider value={{ job, setJob }}>{children}</Ctx.Provider>;
}

export const useJob = () => useContext(Ctx);

export function download(name: string, data: string | Blob, type = "text/plain") {
  const blob = typeof data === "string" ? new Blob([data], { type }) : data;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const fmtTime = (s: number) => (s < 60 ? `${s.toFixed(0)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
export const fmtMm = (mm: number) => (mm >= 1000 ? `${(mm / 1000).toFixed(2)} m` : `${mm.toFixed(0)} mm`);
