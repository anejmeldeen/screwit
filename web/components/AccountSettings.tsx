"use client";

import { useState, type FormEvent } from "react";
import { notifyAccountChange } from "./AccountProvider";
import { api } from "@/lib/api";

export function AccountSettings({ username }: { username: string }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  async function submit(event: FormEvent<HTMLFormElement>, rename: boolean) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const password = String(data.get("current"));
    setBusy(true);
    setMessage("");
    try {
      if (rename) {
        const account = await api.changeUsername(String(data.get("username")), password);
        if (account.username !== username) {
          try {
            const key = "screwit:dashboard:" + username;
            const prefs = localStorage.getItem(key);
            if (prefs) localStorage.setItem("screwit:dashboard:" + account.username, prefs);
            localStorage.removeItem(key);
          } catch { /* Preferences are optional. */ }
          notifyAccountChange();
          window.location.reload();
          return;
        }
        setMessage("Username saved.");
      } else {
        if (data.get("new") !== data.get("confirm")) throw new Error("The new passwords must match.");
        await api.changePassword(password, String(data.get("new")));
        setMessage("Password updated.");
      }
      form.reset();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Could not save account changes.");
    } finally { setBusy(false); }
  }
  return <div className="settings-card">
    {(["username", "password"] as const).map(kind => <details key={kind} className="account-edit">
      <summary>{kind === "username" ? `Change username · ${username}` : "Reset password"}</summary>
      <form onSubmit={event => submit(event, kind === "username")}>
        {kind === "username" && <label>New username<input name="username" autoComplete="username" defaultValue={username}
          pattern={"[A-Za-z0-9_\\-]+"} title="Use letters, numbers, underscores or hyphens" maxLength={32} required /></label>}
        <label>Current password<input name="current" type="password" autoComplete="current-password" minLength={8} maxLength={128} required /></label>
        {kind === "password" && <>
          <label>New password<input name="new" type="password" autoComplete="new-password" minLength={8} maxLength={128} required /></label>
          <label>Confirm new password<input name="confirm" type="password" autoComplete="new-password" minLength={8} maxLength={128} required /></label>
        </>}
        <button className="button-primary" disabled={busy}>{kind === "username" ? "Save username" : "Save password"}</button>
      </form>
    </details>)}
    <p role="status">{message}</p>
  </div>;
}
