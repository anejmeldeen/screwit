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
    <div className="chat-panel">
      <header className="chat-heading"><span className="chat-mark" aria-hidden="true">✳</span><div><h2>Assistant</h2><p>Work on your object.</p></div></header>
      <div ref={scrollRef} className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <h3>What would you like to change?</h3>
            <p>Select a part in the viewer, or ask a question here.</p>
            <div className="chat-suggestions">
              {["Explain the parts", "Help me fix this", "Make a replacement"].map((prompt) => <button type="button" key={prompt} onClick={() => setInput(prompt)}>{prompt}<span aria-hidden="true">↗</span></button>)}
            </div>
          </div>
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
        {error ? <p className="text-sm text-rose-800">{error}</p> : null}
      </div>

      {/* Image attachment chips */}
      {images.length ? (
        <div className="chat-attachments">
          {images.map((u, i) => (
            <span key={i} className="rounded bg-wash px-1.5 py-0.5">
              {u.split("/").pop()}
            </span>
          ))}
        </div>
      ) : null}

      <div className="chat-composer">
        <label className="chat-attach" aria-label="Attach an image">
          <span aria-hidden="true">+</span>
          <input
            type="file"
            accept="image/*"
            multiple
            className="sr-only"
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
          rows={2}
          aria-label="Message the assistant"
          placeholder="Ask a question or describe a change…"
          className="chat-input"
        />
        <button
          data-testid="chat-send"
          onClick={send}
          disabled={streaming || !input.trim()}
          className="chat-send" aria-label="Send message"
        >
          <span aria-hidden="true">↑</span>
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
    <div className={`chat-message ${msg.role === "user" ? "from-user" : "from-assistant"}`}><span className="chat-author">{msg.role === "user" ? "You" : "Assistant"}</span>
      {msg.text ? (
        <div
          data-testid={msg.role === "assistant" ? "assistant-message" : undefined}
          className="chat-bubble"
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
          className="inline-block max-w-[90%] rounded-lg px-3 py-2 text-sm text-muted"
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
