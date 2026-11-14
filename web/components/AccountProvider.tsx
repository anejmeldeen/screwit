"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { api, ApiError, type Account } from "@/lib/api";

const AccountContext = createContext<Account | null>(null);
export const useAccount = () => useContext(AccountContext);

export function AccountProvider({ children }: { children: React.ReactNode }) {
  const [account, setAccount] = useState<Account | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const path = usePathname();
  const router = useRouter();

  useEffect(() => {
    let alive = true;
    const check = () => {
      api.me().then((user) => {
        if (alive) { setAccount(user); setError(false); }
      }).catch((err) => {
        if (!alive) return;
        setAccount(null);
        setError(!(err instanceof ApiError && err.status === 401));
      }).finally(() => { if (alive) setLoading(false); });
    };
    const expire = () => { setAccount(null); setError(false); };
    const sync = (event: StorageEvent) => { if (event.key === "screwit:account-change") check(); };
    check();
    window.addEventListener("focus", check);
    window.addEventListener("storage", sync);
    window.addEventListener("screwit:session-expired", expire);
    return () => {
      alive = false;
      window.removeEventListener("focus", check);
      window.removeEventListener("storage", sync);
      window.removeEventListener("screwit:session-expired", expire);
    };
  }, [attempt]);

  useEffect(() => {
    if (!loading && !error && !account && path !== "/login") {
      router.replace(`/login?next=${encodeURIComponent(path + window.location.search)}`);
    }
  }, [account, error, loading, path, router]);

  if (path === "/login") return <AccountContext.Provider value={account}>{children}</AccountContext.Provider>;
  if (error) return (
    <div className="account-loading" role="alert">
      <p>We couldn&apos;t connect to your account.</p>
      <button className="text-link" onClick={() => { setError(false); setLoading(true); setAttempt((n) => n + 1); }}>Try again</button>
    </div>
  );
  if (loading || !account) return <div className="account-loading" role="status">Opening your workspace…</div>;
  return <AccountContext.Provider key={account.username} value={account}>{children}</AccountContext.Provider>;
}

export function notifyAccountChange() {
  try { localStorage.setItem("screwit:account-change", String(Date.now())); } catch { /* Storage is optional. */ }
}
