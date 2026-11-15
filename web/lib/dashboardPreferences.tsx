"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { useAccount } from "@/components/AccountProvider";

export type WorkspaceSort = "opened" | "name" | "created";
interface Preferences { sort: WorkspaceSort; compact: boolean }
const defaults: Preferences = { sort: "opened", compact: false };
const PreferencesContext = createContext<{ preferences: Preferences; save: (update: Partial<Preferences>) => void; error: string }>({ preferences: defaults, save: () => {}, error: "" });

export function DashboardPreferences({ children }: { children: React.ReactNode }) {
  const account = useAccount();
  const key = `screwit:dashboard:${account?.username}`;
  const [preferences, setPreferences] = useState(defaults);
  const [error, setError] = useState("");
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(key) || "null");
      setPreferences({
        sort: ["opened", "name", "created"].includes(saved?.sort) ? saved.sort : "opened",
        compact: saved?.compact === true,
      });
    } catch { setPreferences(defaults); }
  }, [key]);
  function save(update: Partial<Preferences>) {
    const next = { ...preferences, ...update };
    setPreferences(next);
    try { localStorage.setItem(key, JSON.stringify(next)); setError(""); }
    catch { setError("Your browser couldn't save this preference. It will apply until you reload."); }
  }
  return <PreferencesContext.Provider value={{ preferences, save, error }}>{children}</PreferencesContext.Provider>;
}

export const useDashboardPreferences = () => useContext(PreferencesContext);
