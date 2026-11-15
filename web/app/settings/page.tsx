"use client";

import { AccountSettings } from "@/components/AccountSettings";
import { useAccount } from "@/components/AccountProvider";
import { useDashboardPreferences, type WorkspaceSort } from "@/lib/dashboardPreferences";

export default function SettingsPage() {
  const account = useAccount();
  const { preferences, save, error } = useDashboardPreferences();
  return <>
    <header className="dashboard-heading"><div><p className="dashboard-eyebrow">Make yourself at home</p><h1>Settings.</h1><p>A workspace that works your way.</p></div></header>
    <section className="settings-section" aria-labelledby="account-heading"><h2 id="account-heading">Your account</h2>{account && <AccountSettings username={account.username} />}</section>
    <section className="settings-section" aria-labelledby="preferences-heading"><h2 id="preferences-heading">Dashboard preferences</h2><div className="settings-card">
      <div className="settings-row"><div><label htmlFor="default-sort">Default workspace order</label><p>Choose what comes first when you open your dashboard.</p></div><select id="default-sort" value={preferences.sort} onChange={(event) => save({ sort: event.target.value as WorkspaceSort })}><option value="opened">Last opened</option><option value="name">Name</option><option value="created">Date created</option></select></div>
      <div className="settings-row"><div><label htmlFor="compact-rows">Compact workspace rows</label><p>A little less space between your ideas.</p></div><input id="compact-rows" type="checkbox" checked={preferences.compact} onChange={(event) => save({ compact: event.target.checked })} /></div>
    </div><p className="settings-caption" role="status">{error || "Preferences are saved automatically for this account in this browser."}</p></section>
  </>;
}
