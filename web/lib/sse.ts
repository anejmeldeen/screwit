// SSE client for the two streaming routes (SPEC 4.8 + AGENTS.md decision 2).
//
// Both streams are read with fetch + eventsource-parser, not the browser's
// EventSource, because chat is a POST and (in the token design) headers would
// be needed. Wire format:
//   event: <name>\n
//   data: <json>\n\n
// For chat the data is the event payload WITHOUT the `event` field; for `card`
// the data is {type, payload}. Comment lines (": keep-alive") are ignored by
// the parser. The progress stream emits `event: scan` with a full Scan.

import { createParser, type EventSourceMessage } from "eventsource-parser";
import type { ChatEvent, Scan } from "./contracts.gen";

type CardWire = Extract<ChatEvent, { event?: "card" }>["card"];

// A parsed chat event in the full object shape (ChatEvent) that the app uses.
// decodeChatEvent lifts a wire frame (name + data json) into this shape by
// re-attaching the event discriminator and, for cards, wrapping the payload.
export function decodeChatEvent(
  eventName: string,
  data: unknown,
): ChatEvent | null {
  switch (eventName) {
    case "token":
      return { event: "token", ...(data as { text: string }) };
    case "tool_call":
      return {
        event: "tool_call",
        ...(data as { name: string; args: Record<string, unknown> }),
      };
    case "highlight":
      return { event: "highlight", ...(data as { part_ids: string[] }) };
    case "card":
      // Wire data is {type, payload}; the ChatEvent shape nests it under `card`.
      return { event: "card", card: data as CardWire };
    case "done":
      return { event: "done", ...(data as { message_id: string }) };
    case "error":
      return { event: "error", ...(data as { message: string }) };
    default:
      return null;
  }
}

// Low-level: feed a raw SSE text stream to a callback that receives each
// (eventName, dataString) frame. Exposed for testing against re-encoded
// fixtures. Comment lines and keep-alives are dropped by eventsource-parser.
export function makeSseParser(
  onFrame: (eventName: string, data: string) => void,
) {
  return createParser({
    onEvent: (msg: EventSourceMessage) => {
      // eventsource-parser defaults an eventless `data:` line to "message".
      const name = msg.event || "message";
      onFrame(name, msg.data);
    },
  });
}

// Parse a complete SSE text buffer into decoded ChatEvents (used by tests and
// for buffered responses).
export function parseChatStream(text: string): ChatEvent[] {
  const out: ChatEvent[] = [];
  const parser = makeSseParser((name, data) => {
    let payload: unknown;
    try {
      payload = data ? JSON.parse(data) : {};
    } catch {
      return;
    }
    const ev = decodeChatEvent(name, payload);
    if (ev) out.push(ev);
  });
  parser.feed(text);
  return out;
}

async function* readLines(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  try {
    while (true) {
      if (signal?.aborted) break;
      const { value, done } = await reader.read();
      if (done) break;
      yield decoder.decode(value, { stream: true });
    }
  } finally {
    reader.releaseLock();
  }
}

// Stream a chat POST and invoke onEvent for every decoded ChatEvent.
export async function streamChat(
  scanId: string,
  message: string,
  imageUris: string[],
  onEvent: (ev: ChatEvent) => void | Promise<void>,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`/api/scans/${scanId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ message, image_uris: imageUris }),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`Chat stream failed (${res.status})`);
  }
  let completed = false;
  const events: ChatEvent[] = [];
  const parser = makeSseParser((name, data) => {
    let payload: unknown;
    try {
      payload = data ? JSON.parse(data) : {};
    } catch {
      return;
    }
    const ev = decodeChatEvent(name, payload);
    if (ev?.event === "done" || ev?.event === "error") completed = true;
    if (ev) events.push(ev);
  });
  for await (const chunk of readLines(res.body, signal)) {
    parser.feed(chunk);
    for (const ev of events.splice(0)) await onEvent(ev);
  }
  if (!completed) throw new Error("Screwy’s connection ended before the reply finished. Please try again.");
}

// Subscribe to the progress stream. Emits a full Scan on each `event: scan`.
// Reconnects with exponential backoff and, on reconnect, re-reads GET
// /api/scans/{id} once so no progress is missed while disconnected.
export function subscribeScanEvents(
  scanId: string,
  onScan: (scan: Scan) => void,
  onError?: (err: unknown) => void,
): () => void {
  let stopped = false;
  let controller: AbortController | null = null;
  let backoff = 500;
  const maxBackoff = 10000;

  // Once the scan is terminal (ready or failed) no more progress will come, so
  // stop the stream and its reconnect/resync loop — otherwise it keeps
  // reconnecting and re-reading GET /scans/{id}, which on a slow link steals
  // bandwidth from the viewer's assets during load.
  function isTerminal(scan: Scan): boolean {
    return scan.status === "ready" || scan.status === "failed";
  }

  function deliver(scan: Scan) {
    onScan(scan);
    if (isTerminal(scan)) {
      stopped = true;
      controller?.abort();
      // The terminal `ready` frame can arrive a beat before the final per-stage
      // snapshot (e.g. the `parts` stage still showing running), which left the
      // Parts panel empty until a manual refresh. Do ONE last GET so the client
      // always ends on the fully-complete scan (every stage `done`), which is
      // what flips partsReady and triggers the parts fetch. Fire-and-forget.
      void finalResync();
    }
  }

  async function finalResync() {
    try {
      const res = await fetch(`/api/scans/${scanId}`, {
        headers: { Accept: "application/json" },
      });
      if (res.ok) onScan((await res.json()) as Scan);
    } catch {
      /* best-effort; the terminal frame already delivered the ready status */
    }
  }

  async function resyncOnce() {
    try {
      const res = await fetch(`/api/scans/${scanId}`, {
        headers: { Accept: "application/json" },
      });
      if (res.ok) deliver((await res.json()) as Scan);
    } catch {
      /* best-effort resync; the stream will deliver updates too */
    }
  }

  async function connect(isReconnect: boolean) {
    if (stopped) return;
    if (isReconnect) await resyncOnce();
    if (stopped) return;
    controller = new AbortController();
    try {
      const res = await fetch(`/api/scans/${scanId}/events`, {
        headers: { Accept: "text/event-stream" },
        signal: controller.signal,
      });
      if (!res.ok || !res.body) throw new Error(`events ${res.status}`);
      backoff = 500; // reset after a successful connect
      const parser = makeSseParser((name, data) => {
        if (name !== "scan") return;
        try {
          deliver(JSON.parse(data) as Scan);
        } catch {
          /* ignore malformed frame */
        }
      });
      for await (const chunk of readLines(res.body, controller.signal)) {
        parser.feed(chunk);
      }
    } catch (err) {
      if (stopped) return;
      onError?.(err);
    }
    if (stopped) return;
    // Stream ended or errored: reconnect with backoff.
    const wait = backoff;
    backoff = Math.min(backoff * 2, maxBackoff);
    setTimeout(() => connect(true), wait);
  }

  void connect(false);

  return () => {
    stopped = true;
    controller?.abort();
  };
}
