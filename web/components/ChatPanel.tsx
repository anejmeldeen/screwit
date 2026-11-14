"use client";

import { Suspense, lazy, memo, useEffect, useRef, useState } from "react";
import { streamChat } from "@/lib/sse";
import { markOnce, PERF_MARKS } from "@/lib/perf";
import type { ChatEvent } from "@/lib/contracts.gen";
import { PartText } from "@/components/cards/PartText";
import { Spinner } from "@/components/ui";

// Markdown (react-markdown + remark-gfm + micromark, ~49 KB) and the rich cards
// load ONLY when an assistant message actually needs them, so they stay out of
// the workspace shell bundle.
const Markdown = lazy(() =>
  import("@/components/MarkdownMessage").then((m) => ({ default: m.MarkdownMessage })),
);
const ChatCards = lazy(() =>
  import("@/components/cards/ChatCards").then((m) => ({ default: m.ChatCards })),
);

type Card = { type: string; payload: unknown };

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  cards: Card[];
}

export function ChatPanel({
  scanId,
  onHighlight,
  onPartChip,
  onShowCad,
}: {
  scanId: string;
  onHighlight: (ids: string[]) => void;
  onPartChip: (id: string) => void;
  onShowCad: (cad: Record<string, unknown>) => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [images, setImages] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || streaming) return;
    setError(null);
    const userMsg: Message = {
      id: `u_${Date.now()}`,
      role: "user",
      text,
      cards: [],
    };
    const assistantId = `a_${Date.now()}`;
    setMessages((m) => [
      ...m,
      userMsg,
      { id: assistantId, role: "assistant", text: "", cards: [] },
    ]);
    setInput("");
    setStreaming(true);

    const sentImages = images;
    setImages([]);

    try {
      let firstToken = true;
      // Coalesce rapid token updates into one state update per animation frame
      // so the transcript does not re-render on every single streamed token.
      let pendingText = "";
      let rafQueued = false;
      const flush = () => {
        rafQueued = false;
        if (!pendingText) return;
        const chunk = pendingText;
        pendingText = "";
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId ? { ...msg, text: msg.text + chunk } : msg,
          ),
        );
      };

      await streamChat(
        scanId,
        text,
        sentImages,
        (ev: ChatEvent) => {
          switch (ev.event) {
            case "token": {
              if (firstToken) {
                markOnce(PERF_MARKS.chatFirstToken);
                firstToken = false;
              }
              pendingText += (ev as { text: string }).text;
              if (!rafQueued) {
                rafQueued = true;
                requestAnimationFrame(flush);
              }
              break;
            }
            case "highlight":
              onHighlight((ev as { part_ids: string[] }).part_ids);
              break;
            case "card": {
              const card = (ev as { card: unknown }).card as Card;
              setMessages((m) =>
                m.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, cards: [...msg.cards, card] }
                    : msg,
                ),
              );
              break;
            }
            case "done":
              flush();
              break;
            case "tool_call":
            case "error":
            default:
              break;
          }
        },
      );
      flush();
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setStreaming(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {messages.length === 0 ? (
          <p className="text-sm text-slate-500">
            Ask about the object — &ldquo;what is this?&rdquo;, &ldquo;how does
            it work?&rdquo;, &ldquo;why might it be failing?&rdquo;, or
            &ldquo;make me a replacement part&rdquo;.
          </p>
        ) : null}
        {messages.map((msg) => (
          <MessageRow
            key={msg.id}
            msg={msg}
            onPartChip={onPartChip}
            onShowCad={onShowCad}
          />
        ))}
        {streaming ? (
          <div className="pl-1">
            <Spinner label="thinking…" />
          </div>
        ) : null}
        {error ? <p className="text-sm text-rose-300">{error}</p> : null}
      </div>

      {/* Image attachment chips */}
      {images.length ? (
        <div className="flex flex-wrap gap-1 border-t border-slate-800 px-3 py-1 text-xs text-slate-400">
          {images.map((u, i) => (
            <span key={i} className="rounded bg-slate-800 px-1.5 py-0.5">
              {u.split("/").pop()}
            </span>
          ))}
        </div>
      ) : null}

      <div className="flex items-end gap-2 border-t border-slate-800 p-2">
        <label className="cursor-pointer rounded border border-slate-700 px-2 py-2 text-sm hover:border-slate-500">
          📎
          <input
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => {
              // Attach by object URL placeholder; a full build uploads via the
              // storage client and sends the returned keys.
              const files = Array.from(e.target.files ?? []);
              setImages((cur) => [
                ...cur,
                ...files.map((f) => `attachments/${f.name}`),
              ]);
            }}
          />
        </label>
        <textarea
          data-testid="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          rows={1}
          placeholder="Ask a question…"
          className="min-h-[2.5rem] flex-1 resize-none rounded border border-slate-700 bg-ink px-3 py-2 text-sm"
        />
        <button
          data-testid="chat-send"
          onClick={send}
          disabled={streaming || !input.trim()}
          className="rounded bg-accent px-3 py-2 text-sm font-medium text-ink disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  );
}

// One message row, memoized so only the message that changed re-renders while a
// response streams in (lowers main-thread work per token).
const MessageRow = memo(function MessageRow({
  msg,
  onPartChip,
  onShowCad,
}: {
  msg: Message;
  onPartChip: (id: string) => void;
  onShowCad: (cad: Record<string, unknown>) => void;
}) {
  return (
    <div className={msg.role === "user" ? "text-right" : "text-left"}>
      {msg.text ? (
        <div
          data-testid={msg.role === "assistant" ? "assistant-message" : undefined}
          className={`inline-block max-w-[90%] rounded-lg px-3 py-2 text-sm ${
            msg.role === "user"
              ? "bg-accent text-ink"
              : "bg-slate-800 text-slate-100"
          }`}
        >
          {msg.role === "assistant" ? (
            <Suspense
              fallback={<span className="whitespace-pre-wrap">{msg.text}</span>}
            >
              <Markdown text={msg.text} onPartChip={onPartChip} />
            </Suspense>
          ) : (
            msg.text
          )}
        </div>
      ) : msg.role === "assistant" ? (
        // Empty assistant bubble before the first token: keep the testid present
        // but let the perf suite see text appear once streaming starts.
        <div
          data-testid="assistant-message"
          className="inline-block max-w-[90%] rounded-lg px-3 py-2 text-sm text-slate-500"
        />
      ) : null}
      {msg.cards.length ? (
        <Suspense fallback={null}>
          <ChatCards cards={msg.cards} onPartChip={onPartChip} onShowCad={onShowCad} />
        </Suspense>
      ) : null}
    </div>
  );
});
