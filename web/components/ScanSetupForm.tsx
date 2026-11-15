"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { uploadToPresignedPost } from "@/lib/storage";
import type { CreateScanRequest, CreateScanResponse } from "@/lib/contracts.gen";
import { ErrorNote, Spinner } from "@/components/ui";

const MIN_DURATION_S = 10;
const MAX_DURATION_S = 90;
const REC_MIN_S = 20;
const REC_MAX_S = 60;
const MAX_BYTES = 500 * 1024 * 1024; // 500 MB

type Phase = "idle" | "checking" | "uploading" | "starting" | "done" | "error";

// Read a video file's duration client-side (courtesy pre-check only).
function probeDurationS(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    if (!file.type.startsWith("video/")) return resolve(null);
    const url = URL.createObjectURL(file);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      resolve(Number.isFinite(v.duration) ? v.duration : null);
    };
    v.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(null);
    };
    v.src = url;
  });
}

export function ScanSetupForm({ title, onTitleChange, onStarted }: {
  title: string;
  onTitleChange: (title: string) => void;
  onStarted: (scanId: string) => void;
}) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const pending = useRef<{ key: string; scan: CreateScanResponse } | null>(null);
  function chooseFiles(files: FileList | File[]) {
    setSelectedFiles(Array.from(files)); setPhase("idle"); setError(null); setProgress(0);
  }

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      setError(null);
      const list = Array.from(files);
      if (list.length === 0 || !title.trim()) return;
      const allPhotos = list.every(f => f.type.startsWith("image/"));
      if ((!allPhotos && (list.length !== 1 || !list[0].type.startsWith("video/"))) || (allPhotos && list.length > 60)) {
        setPhase("error");
        setError("Choose one video or up to 60 photos. Upload videos and photos separately.");
        return;
      }

      const isPhotos = list.every((f) => f.type.startsWith("image/"));
      const first = list[0];

      // Courtesy pre-checks; the server re-validates everything.
      setPhase("checking");
      const totalBytes = list.reduce((n, f) => n + f.size, 0);
      if (totalBytes > MAX_BYTES) {
        setPhase("error");
        setError(
          `That is ${(totalBytes / 1024 / 1024).toFixed(0)} MB; the limit is 500 MB. Record a shorter clip.`,
        );
        return;
      }
      if (!isPhotos) {
        const dur = await probeDurationS(first);
        if (dur != null && (dur < MIN_DURATION_S || dur > MAX_DURATION_S)) {
          setPhase("error");
          setError(
            `Video is ${dur.toFixed(0)} s; please keep it between ${MIN_DURATION_S} and ${MAX_DURATION_S} s (${REC_MIN_S}-${REC_MAX_S} s works best).`,
          );
          return;
        }
      }

      try {
        const body: CreateScanRequest & { title: string } = {
          title: title.trim(),
          input_type: isPhotos ? "photos" : "video",
          ...(isPhotos ? { photo_count: list.length } : {}),
        } as CreateScanRequest & { title: string };
        const key = JSON.stringify([body, list.map(f => [f.name, f.size, f.lastModified])]);
        const created = pending.current?.key === key ? pending.current.scan : await api.createScan(body);
        pending.current = { key, scan: created };

        setPhase("uploading");
        // One presigned POST per file (photos may have several).
        const uploads = created.uploads;
        for (let i = 0; i < uploads.length; i++) {
          const file = list[Math.min(i, list.length - 1)];
          await uploadToPresignedPost(uploads[i], file, (p) => {
            const base = (i / uploads.length) * 100;
            setProgress(base + (p.loaded / p.total) * (100 / uploads.length));
          });
        }

        setPhase("starting");
        await api.startScan(created.scan_id);
        setPhase("done");
        onStarted(created.scan_id);
      } catch (e) {
        setPhase("error");
        const msg =
          e instanceof ApiError
            ? `${e.message} (${e.status})`
            : String((e as Error)?.message ?? e);
        setError(msg);
      }
    },
    [title, onStarted],
  );

  const busy =
    phase === "checking" ||
    phase === "uploading" ||
    phase === "starting" ||
    phase === "done";

  return (
    <form className="workspace-upload-form" onSubmit={e => {
      e.preventDefault(); if (!busy) void handleFiles(selectedFiles);
    }}>
      <h2 className="text-2xl font-bold">What are we building?</h2>
      <p className="mt-1 text-ink">
        Upload a video or photos of your object.
      </p>

      <label className="workspace-name-field" htmlFor="workspace-title">Workspace title
        <input id="workspace-title" name="title" placeholder="e.g. Kitchen tap repair" required maxLength={120}
          value={title} onChange={e => onTitleChange(e.target.value)} disabled={busy} autoComplete="off" />
      </label>

      {/* Capture / upload control */}
      <div
        className={`upload-drop mt-8 rounded-lg border-2 border-dashed text-center transition ${
          dragOver ? "border-accent bg-sky-100" : "border-line"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          if (!busy && e.dataTransfer.files?.length) chooseFiles(e.dataTransfer.files);
        }}
      >
        <p className="text-ink">Drag and drop a video or photos here</p>
        <p className="my-2 text-sm text-muted">or</p>

        <div className="flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
          {/* Phone-preferred path: opens the native camera on most phones.
              `capture` is only a hint, so the ordinary picker is always offered too. */}
          <label className="focus-within:outline focus-within:outline-2 focus-within:outline-offset-4 focus-within:outline-ink cursor-pointer rounded-md bg-accent px-4 py-2 font-medium text-paper hover:bg-ink/90">
            Record with camera
            <input
              type="file"
              accept="video/*"
              capture="environment"
              className="sr-only"
              disabled={busy}
              onChange={(e) => e.target.files && chooseFiles(e.target.files)}
            />
          </label>

          <label className="focus-within:outline focus-within:outline-2 focus-within:outline-offset-4 focus-within:outline-ink cursor-pointer rounded-md border border-line px-4 py-2 font-medium hover:border-line">
            Choose a file
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*,image/*"
              multiple
              className="sr-only"
              disabled={busy}
              onChange={(e) => e.target.files && chooseFiles(e.target.files)}
            />
          </label>
        </div>

        {busy ? (
          <div className="mt-6">
            <Spinner
              label={
                phase === "uploading"
                  ? `Uploading ${progress.toFixed(0)}%`
                  : phase === "starting"
                    ? "Preparing your scan…"
                    : phase === "done"
                      ? "Opening workbench…"
                      : "Checking file…"
              }
            />
            {phase === "uploading" ? (
              <div role="progressbar" aria-label="Upload progress" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100} className="mx-auto mt-2 h-1.5 w-full max-w-64 overflow-hidden rounded bg-wash">
                <div
                  className="h-full bg-accent transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            ) : null}
          </div>
        ) : null}
      </div>

      {selectedFiles.length > 0 ? <div className="workspace-file-summary">
        <p>{selectedFiles.length === 1 ? selectedFiles[0].name : `${selectedFiles.length} photos selected`} <span>· {(selectedFiles.reduce((n, f) => n + f.size, 0) / 1024 / 1024).toFixed(1)} MB</span></p>
        <button type="button" disabled={busy} onClick={() => { chooseFiles([]); if (fileInputRef.current) fileInputRef.current.value = ""; }}>Remove files</button>
      </div> : null}
      <button type="submit" className="workspace-generate" disabled={busy || selectedFiles.length === 0 || !title.trim()}>
        {busy ? "Preparing your workspace…" : phase === "error" ? "Try again" : "Generate workspace"}
      </button>
      <p className="workspace-setup-note">Your model and detected parts will appear in this editor as processing completes.</p>

      {error ? (
        <div className="mt-4">
          <ErrorNote message={error} />
        </div>
      ) : null}

      {/* Scale board + capture tips */}
      <section className="mt-8 grid gap-6 md:grid-cols-2">
        <div className="rounded-lg border border-line bg-panel p-5">
          <h2 className="font-semibold">Scale board</h2>
          <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-ink">
            <li>
              Print the board PDF at <strong>100% scale</strong> (no
              &ldquo;fit to page&rdquo;).
            </li>
            <li>
              Check one printed square with a ruler before using it — a board
              shown on a screen has an unknown size and is not accepted.
            </li>
            <li>
              Lay it <strong>flat</strong> on the same surface the object stands
              on, right beside the object, arrow pointing away from your first
              camera position.
            </li>
          </ol>
          <a
            href="/scale-board.pdf"
            download
            className="mt-3 inline-flex rounded-md border border-line px-3 py-1.5 text-sm hover:border-line"
          >
            Download scale board (A4 PDF)
          </a>
          <p className="mt-2 text-xs text-muted">
            No board? Put a credit card (85.60 × 53.98 mm) in frame and set the
            scale later in the viewer.
          </p>
        </div>

        <div className="rounded-lg border border-line bg-panel p-5">
          <h2 className="font-semibold">Capture tips</h2>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-ink">
            <li>Record a slow orbit, {REC_MIN_S}-{REC_MAX_S} s (10-90 s accepted).</li>
            <li>Keep the whole object and the board in frame.</li>
            <li>Even, diffuse light; avoid harsh shadows and glare.</li>
            <li>
              <strong>iPhone:</strong> turn <em>HDR video</em> off in Camera
              settings for cleaner frames.
            </li>
            <li>File size limit is 500 MB.</li>
          </ul>
        </div>
      </section>

      <WebcamRecorder onRecorded={chooseFiles} disabled={busy} />
    </form>
  );
}

// Optional desktop webcam recorder (MediaRecorder). Hidden/disabled where the
// API is unavailable; this is a courtesy path for laptops.
function WebcamRecorder({
  onRecorded,
  disabled,
}: {
  onRecorded: (files: File[]) => void;
  disabled: boolean;
}) {
  const [recording, setRecording] = useState(false);
  const [supported] = useState(
    () =>
      typeof window !== "undefined" &&
      typeof window.MediaRecorder !== "undefined" &&
      !!navigator.mediaDevices?.getUserMedia,
  );
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => () => {
    if (recRef.current) { recRef.current.onstop = null; if (recRef.current.state !== "inactive") recRef.current.stop(); }
    streamRef.current?.getTracks().forEach(track => track.stop());
  }, []);

  if (!supported) return null;

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      chunksRef.current = [];
      const rec = new MediaRecorder(stream);
      rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      rec.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: rec.mimeType || "video/webm",
        });
        const ext = (rec.mimeType || "video/webm").includes("mp4")
          ? "mp4"
          : "webm";
        const file = new File([blob], `recording.${ext}`, { type: blob.type });
        streamRef.current?.getTracks().forEach((t) => t.stop());
        onRecorded([file]);
      };
      recRef.current = rec;
      rec.start();
      setRecording(true);
    } catch {
      /* permission denied; the file picker paths still work */
    }
  }

  function stop() {
    recRef.current?.stop();
    setRecording(false);
  }

  return (
    <section className="mt-6 rounded-lg border border-line bg-panel p-5">
      <h2 className="font-semibold">Record from a webcam (optional)</h2>
      <video
        ref={videoRef}
        muted
        playsInline
        className={`mt-3 w-full max-w-sm rounded ${recording ? "block" : "hidden"}`}
      />
      <div className="mt-3">
        {recording ? (
          <button
            type="button" onClick={stop}
            className="rounded-md bg-rose-600 px-4 py-2 font-medium text-white hover:bg-rose-500"
          >
            Stop and use recording
          </button>
        ) : (
          <button
            type="button" onClick={start}
            disabled={disabled}
            className="rounded-md border border-line px-4 py-2 font-medium hover:border-line disabled:opacity-50"
          >
            Start webcam recording
          </button>
        )}
      </div>
    </section>
  );
}
