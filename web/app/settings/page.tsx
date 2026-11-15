"use client";

import { AccountSettings } from "@/components/AccountSettings";
import { useAccount } from "@/components/AccountProvider";

export default function SettingsPage() {
  const account = useAccount();
  return <>
    <header className="workspace-section-heading"><h1>Settings</h1></header>
    <section className="settings-section" aria-labelledby="account-heading"><h2 id="account-heading">Your account</h2>{account && <AccountSettings username={account.username} />}</section>
    <section className="settings-section" aria-labelledby="connections-heading">
      <h2 id="connections-heading">Connections</h2>
      <div className="settings-card">
        {["Google", "LinkedIn", "GitHub"].map(provider => (
          <div className="settings-row" key={provider}>
            <h3>{provider}</h3>
            <button type="button" className="connection-preview" aria-label={`Connect ${provider}`}>Connect</button>
          </div>
        ))}
      </div>
    </section>
  </>;
}
