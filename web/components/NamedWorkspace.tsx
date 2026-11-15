"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { workspaceSlugs } from "@/lib/workspacePaths";
import { Workspace } from "./Workspace";

export function NamedWorkspace({ segment, partId }: { segment: string; partId?: string }) {
  const router = useRouter();
  const [build, setBuild] = useState<{ scan_id: string; title: string } | null>(null);
  const [message, setMessage] = useState("");
  useEffect(() => {
    let active = true;
    api.builds().then(builds => {
      if (!active) return;
      const slugs = workspaceSlugs(builds);
      const match = builds.find(b => b.scan_id === segment || slugs.get(b.scan_id) === segment);
      if (!match) { setMessage("Workspace not found."); return; }
      const slug = slugs.get(match.scan_id)!;
      if (segment !== slug) {
        router.replace(`/scan/${encodeURIComponent(slug)}${partId ? "/part/" + encodeURIComponent(partId) : ""}${window.location.search}${window.location.hash}`);
      } else setBuild(match);
    }).catch(() => { if (active) setMessage("Could not load this workspace. Refresh to retry."); });
    return () => { active = false; };
  }, [segment, partId, router]);
  if (message) return <div className="account-loading is-editor"><p role="alert">{message}</p><Link href="/">Workspaces</Link></div>;
  if (!build) return <div className="account-loading is-editor" role="status">Opening workspace…</div>;
  return <Workspace scanId={build.scan_id} title={build.title} initialPartId={partId} />;
}
