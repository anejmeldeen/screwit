"use client";

import { useEffect, useState, type FormEvent } from "react";
import { api } from "@/lib/api";
import { notifyAccountChange } from "@/components/AccountProvider";

export default function LoginPage() {
  const [register, setRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const legacy = new URLSearchParams(window.location.search).get("next");
    if (legacy !== null) {
      try { sessionStorage.setItem("screwit:return-to", legacy); } catch { /* Default to home. */ }
      window.history.replaceState(window.history.state, "", "/login");
    }
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await (register ? api.register(username, password) : api.login(username, password));
      notifyAccountChange();
      let next = "/";
      try { next = sessionStorage.getItem("screwit:return-to") || "/"; sessionStorage.removeItem("screwit:return-to"); } catch { /* Default to home. */ }
      // Only return to a path in this app, never a foreign origin or the login form.
      let target = new URL("/", window.location.origin);
      try { target = new URL(next, window.location.origin); } catch { /* Use home for malformed legacy links. */ }
      window.location.replace(target.origin === window.location.origin && target.pathname !== "/login" ? target.href : "/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to log in. Please try again.");
      setBusy(false);
    }
  }

  return (
    <div className="home-page login-page">
      <section className="login-content" aria-labelledby="login-title">
        <div className="hero-copy">
          <h1 id="login-title">Screw It.</h1>
          <p className="hero-subtitle">Let&apos;s build.</p>
        </div>
        <form className="login-form" onSubmit={submit} aria-busy={busy}>
          <h2>{register ? "Make it yours." : "Welcome back."}</h2>
          <p>{register ? "Create an account for your builds." : "Log in to your builds."}</p>
          <label htmlFor="username">Username</label>
          <input id="username" name="username" autoComplete="username" autoCapitalize="none" spellCheck={false}
            required maxLength={32} pattern="[A-Za-z0-9_\-]+" title="Letters, numbers, underscores and hyphens only"
            value={username} onChange={(event) => setUsername(event.target.value)} disabled={busy} />
          <label htmlFor="password">Password</label>
          <input id="password" name="password" type="password" autoComplete={register ? "new-password" : "current-password"}
            required minLength={register ? 8 : undefined} maxLength={128}
            aria-describedby={register ? "password-hint" : undefined}
            value={password} onChange={(event) => setPassword(event.target.value)} disabled={busy} />
          {register && <small id="password-hint">At least 8 characters.</small>}
          {error && <p className="login-error" role="alert">{error}</p>}
          <button className="button-primary" type="submit" disabled={busy}>
            {busy ? "One moment…" : register ? "Create account" : "Log in"}<span aria-hidden="true">↗</span>
          </button>
          <button className="login-switch text-link" type="button" disabled={busy}
            onClick={() => { setRegister(!register); setError(""); setPassword(""); }}>
            {register ? "Already have an account? Log in" : "New here? Create an account"}
          </button>
        </form>
      </section>
      <footer className="site-footer"><span>HackUMass 2026</span></footer>
    </div>
  );
}
