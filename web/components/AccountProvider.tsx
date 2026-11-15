"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { api, ApiError, type Account } from "@/lib/api";
import { clearCachedBuilds } from "@/lib/buildsCache";

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
        const signedOut = err instanceof ApiError && err.status === 401;
        if (signedOut) clearCachedBuilds();
        setAccount(null);
        setError(!signedOut);
      }).finally(() => { if (alive) setLoading(false); });
    };
    const expire = () => { clearCachedBuilds(); setAccount(null); setError(false); };
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
      try { sessionStorage.setItem("screwit:return-to", path + window.location.search + window.location.hash); } catch { /* Default to home. */ }
      router.replace("/login");
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

// Called after signing in, signing out or renaming the account: other tabs re-check the account,
// and this tab drops the cached workspace list so the next account never sees the previous one's.
export function notifyAccountChange() {
  clearCachedBuilds();
  try { localStorage.setItem("screwit:account-change", String(Date.now())); } catch { /* Storage is optional. */ }
}
