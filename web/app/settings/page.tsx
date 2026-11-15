"use client";

import { AccountSettings } from "@/components/AccountSettings";
import { useAccount } from "@/components/AccountProvider";
import { connectionLogos } from "@/components/ConnectionLogos";

export default function SettingsPage() {
  const account = useAccount();
  return <>
    <header className="workspace-section-heading"><h1>Your Settings</h1></header>
    {account && <AccountSettings username={account.username} />}
    <section className="settings-section" aria-labelledby="connections-heading">
      <div className="settings-card">
        <header className="settings-card-head">
          <h2 id="connections-heading">Connections</h2>
        </header>
        {(["Google", "LinkedIn", "GitHub"] as const).map(provider => (
          <div className="settings-row connection-row" key={provider}>
            <span className="connection-logo">{connectionLogos[provider]}</span>
            <div><h3>{provider}</h3><p>Not connected</p></div>
            <button type="button" className="connection-preview" aria-label={`Connect ${provider}`}>Connect</button>
          </div>
        ))}
      </div>
    </section>
  </>;
}
