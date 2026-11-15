"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { Workspace } from "@/components/Workspace";
import { ScanSetupForm } from "@/components/ScanSetupForm";

export default function NewWorkspacePage() {
  const router = useRouter();
  const [title, setTitle] = useState("");
  const onStarted = useCallback((scanId: string) => router.replace(`/scan/${scanId}`), [router]);
  return <Workspace scanId="" title={title.trim() || "New workspace"}
    creationForm={<ScanSetupForm title={title} onTitleChange={setTitle} onStarted={onStarted} />} />;
}
