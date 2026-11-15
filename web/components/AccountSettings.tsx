"use client";

import { useState, type ChangeEvent, type FormEvent } from "react";
import { notifyAccountChange } from "./AccountProvider";
import { PasswordRequirements, passwordRules } from "./PasswordRequirements";
import { api } from "@/lib/api";

type Note = { text: string; error: boolean } | null;

// Username and password are two separate settings sections, each with its own form, save button
// and message, laid out like the Connections card below them.
export function AccountSettings({ username }: { username: string }) {
  return <>
    <UsernameSection username={username} />
    <PasswordSection username={username} />
  </>;
}

function UsernameSection({ username }: { username: string }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note>(null);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    setBusy(true);
    setNote(null);
    try {
      const account = await api.changeUsername(String(data.get("username")), String(data.get("current")));
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
      form.reset();
      setNote({ text: "Username saved.", error: false });
    } catch (err) {
      setNote({ text: err instanceof Error ? err.message : "Could not change the username.", error: true });
    } finally { setBusy(false); }
  }
  return <section className="settings-section" aria-labelledby="username-heading">
    <form className="settings-card settings-form" onSubmit={submit}>
      <header className="settings-card-head"><h2 id="username-heading">Username</h2></header>
      <div className="settings-fields">
        <label><span>New username</span>
          <input name="username" autoComplete="username" defaultValue={username} pattern={"[A-Za-z0-9_\\-]+"}
            title="Use letters, numbers, underscores or hyphens" maxLength={32} required /></label>
        <label><span>Current password</span>
          <input name="current" type="password" autoComplete="current-password" minLength={8} maxLength={128} required /></label>
      </div>
      <div className="settings-actions">
        <p role="status" className={note?.error ? "is-error" : undefined}>{note?.text}</p>
        <button className="button-primary" disabled={busy}>{busy ? "Saving…" : "Save username"}</button>
      </div>
    </form>
  </section>;
}

const emptyPasswords = { current: "", next: "", confirm: "" };

function PasswordSection({ username }: { username: string }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<Note>(null);
  const [values, setValues] = useState(emptyPasswords);
  const rules = passwordRules({ password: values.next, username, current: values.current, confirm: values.confirm });
  const field = (key: keyof typeof emptyPasswords) => ({
    value: values[key],
    onChange: (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.target.value;
      setValues(current => ({ ...current, [key]: value }));
    },
  });
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setNote(null);
    const unmet = rules.find(rule => !rule.met);
    if (unmet) {
      setNote({ text: unmet.problem, error: true });
      return;
    }
    setBusy(true);
    try {
      await api.changePassword(values.current, values.next);
      setValues(emptyPasswords);
      // The server ends every other session when the password changes.
      setNote({ text: "Password updated. Other devices are signed out.", error: false });
    } catch (err) {
      setNote({ text: err instanceof Error ? err.message : "Could not change the password.", error: true });
    } finally { setBusy(false); }
  }
  return <section className="settings-section" aria-labelledby="password-heading">
    <form className="settings-card settings-form" onSubmit={submit}>
      <header className="settings-card-head"><h2 id="password-heading">Password</h2></header>
      <div className="settings-fields">
        <label className="is-full"><span>Current password</span>
          <input name="current" type="password" autoComplete="current-password" minLength={8} maxLength={128} required
            {...field("current")} /></label>
        <label><span>New password</span>
          <input name="new" type="password" autoComplete="new-password" minLength={8} maxLength={128} required
            aria-describedby="new-password-rules" {...field("next")} /></label>
        <label><span>Confirm new password</span>
          <input name="confirm" type="password" autoComplete="new-password" minLength={8} maxLength={128} required
            aria-describedby="new-password-rules" {...field("confirm")} /></label>
        <PasswordRequirements id="new-password-rules" rules={rules} />
      </div>
      <div className="settings-actions">
        <p role="status" className={note?.error ? "is-error" : undefined}>{note?.text}</p>
        <button className="button-primary" disabled={busy}>{busy ? "Saving…" : "Update password"}</button>
      </div>
    </form>
  </section>;
}
