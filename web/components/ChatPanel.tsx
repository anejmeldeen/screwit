"use client";

import { Suspense, lazy, memo, useEffect, useRef, useState } from "react";
import { streamChat } from "@/lib/sse";
import { markOnce, PERF_MARKS } from "@/lib/perf";
import type { ChatEvent } from "@/lib/contracts.gen";
import { PartText } from "@/components/cards/PartText";
import { ScrewyIcon } from "@/components/ScrewyIcon";
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
  onPartsChanged,
  selectedPartId,
}: {
  scanId: string;
  selectedPartId?: string | null;
  onHighlight: (ids: string[]) => void;
  onPartChip: (id: string) => void;
  onShowCad: (cad: Record<string, unknown>) => void;
  // Called once after a reply in which Screwy changed the workspace's parts (the add_part tool).
  onPartsChanged?: () => void;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [progress, setProgress] = useState("Screwy is thinking…");
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
    setProgress("Screwy is thinking…");
    setStreaming(true);

    const sentImages = images;
    setImages([]);

    // An add_part tool call means the workspace has a new part. Refresh the parts once the reply
    // has finished (on `done`, or when the stream ends some other way), and only once.
    let fittingEars = false;
    let partsChanged = false;
    let partsRefreshed = false;
    const refreshPartsOnce = () => {
      if (!partsChanged || partsRefreshed) return;
      partsRefreshed = true;
      onPartsChanged?.();
    };

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
        selectedPartId
          ? text.replace(/\b(?:this part|that part|the selected (?:part|piece)|this piece)\b/gi, () => `[[part:${selectedPartId}]]`)
          : text,
        sentImages,
        async (ev: ChatEvent) => {
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
              refreshPartsOnce();
              break;
            case "error":
              flush();
              setError("Screwy couldn’t finish that reply. Please try again. " +
                (ev as { message: string }).message);
              break;
            case "tool_call":
              if (ev.name === "fit_headphone_ears") {
                fittingEars = true;
                setProgress("Preparing CAD preview…");
                break;
              }
              if (ev.name === "add_part") partsChanged = true;
              if (fittingEars) break;
              if (ev.name === "add_part" || ev.name === "create_cad") {
                setProgress(ev.name === "create_cad" ? "Generating the CAD model…" : "Generating the fitted model…");
                // Let the status paint even when a local preset finishes in a single SSE chunk.
                await new Promise(resolve => setTimeout(resolve, 1000));
              }
              break;
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
      refreshPartsOnce();
    }
  }

  return (
    <div className="chat-panel">
      <header className="chat-heading"><span className="chat-mark screwy-avatar"><ScrewyIcon /></span><div><h2>Screwy</h2><p>Let’s build.</p></div></header>
      <div ref={scrollRef} className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <h3>Hi, I’m Screwy.</h3>
            <p>What would you like to change? Select a part or tell me what you have in mind.</p>

          </div>
        ) : null}
        {messages.map((msg) => (
          <MessageRow
            key={msg.id}
            msg={msg}
            progress={progress}
            loading={streaming && msg.id === messages[messages.length - 1]?.id}
            onPartChip={onPartChip}
            onShowCad={onShowCad}
          />
        ))}
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

      {messages.length === 0 && <div className="chat-suggestions">
              {["Explain the parts", "Help me fix this", "Make a replacement"].map((prompt) => <button type="button" key={prompt} onClick={() => setInput(prompt)}>{prompt}</button>)}
            </div>}
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
          aria-label="Message Screwy"
          placeholder="Ask Screwy…"
          className="chat-input"
        />
        <button
          data-testid="chat-send"
          onClick={send}
          disabled={streaming || !input.trim()}
          className="chat-send" aria-label="Send message"
        >
          <svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m21 3-7 18-4-7-7-4 18-7Z" /><path d="m10 14 5-5" /></svg>
        </button>
      </div>
    </div>
  );
}

// One message row, memoized so only the message that changed re-renders while a
// response streams in (lowers main-thread work per token).
const MessageRow = memo(function MessageRow({
  msg,
  progress,
  loading,
  onPartChip,
  onShowCad,
}: {
  msg: Message;
  progress: string;
  loading: boolean;
  onPartChip: (id: string) => void;
  onShowCad: (cad: Record<string, unknown>) => void;
}) {
  // Older running backends may still stream the retired provider prefix.
  // Strip it from the accumulated text, including while the prefix is incomplete.
  const text = msg.role === "assistant"
    ? msg.text.replace(/^\s*\[online:(?:[^\]]*\]\s*|[^\]]*$)/i, "")
    : msg.text;
  return (
    <div className={`chat-message ${msg.role === "user" ? "from-user" : "from-assistant"}`}><span className="chat-author">{msg.role === "assistant" ? <span className="screwy-message-avatar"><ScrewyIcon /></span> : null}{msg.role === "user" ? "You" : "Screwy"}</span>
      {text ? (
        <div
          data-testid={msg.role === "assistant" ? "assistant-message" : undefined}
          className="chat-bubble"
        >
          {msg.role === "assistant" ? (
            <Suspense
              fallback={<span className="whitespace-pre-wrap">{text}</span>}
            >
              <Markdown text={text} onPartChip={onPartChip} />
            </Suspense>
          ) : (
            text
          )}
        </div>
      ) : msg.role === "assistant" ? (
        // Empty assistant bubble before the first token: keep the testid present
        // but let the perf suite see text appear once streaming starts.
        <div
          data-testid="assistant-message"
          className="chat-pending-message"
        />
      ) : null}
      {loading ? <div className="mt-1"><Spinner label={progress} /></div> : null}
      {msg.cards.length ? (
        <Suspense fallback={null}>
          <ChatCards cards={msg.cards} onPartChip={onPartChip} onShowCad={onShowCad} />
        </Suspense>
      ) : null}
    </div>
  );
});
