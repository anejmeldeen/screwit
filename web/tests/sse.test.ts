import { describe, it, expect } from "vitest";
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseChatStream, decodeChatEvent } from "../lib/sse";

const scansDir = resolve(__dirname, "..", "..", "fixtures", "scans");

// A fixture chat_script event -> the on-the-wire `data` JSON payload (the event
// object WITHOUT its `event` field; for `card` the data is {type, payload}).
function wireData(ev: Record<string, unknown>): unknown {
  const name = ev.event as string;
  switch (name) {
    case "token":
      return { text: ev.text };
    case "tool_call":
      return { name: ev.name, args: ev.args };
    case "highlight":
      return { part_ids: ev.part_ids };
    case "card":
      // Fixture nests the card under `card`; the wire data is that object.
      return ev.card;
    case "done":
      return { message_id: ev.message_id };
    case "error":
      return { message: ev.message };
    default:
      throw new Error(`unknown fixture event: ${name}`);
  }
}

// Encode a list of fixture events into one SSE wire buffer, inserting the
// server's ": keep-alive" comment lines that the parser must ignore.
function encodeWire(events: Record<string, unknown>[]): string {
  const lines: string[] = [];
  for (const ev of events) {
    lines.push(": keep-alive"); // must be ignored
    lines.push(`event: ${ev.event}`);
    lines.push(`data: ${JSON.stringify(wireData(ev))}`);
    lines.push(""); // blank line terminates the frame
  }
  lines.push("");
  return lines.join("\n");
}

function allChatScripts(): { scanId: string; events: Record<string, unknown>[] }[] {
  const out: { scanId: string; events: Record<string, unknown>[] }[] = [];
  for (const scanId of readdirSync(scansDir)) {
    const file = join(scansDir, scanId, "chat_script.json");
    if (!existsSync(file)) continue;
    const script = JSON.parse(readFileSync(file, "utf8"));
    for (const turn of script.turns) {
      out.push({ scanId, events: turn.events });
    }
  }
  return out;
}

describe("SSE parser over every fixture chat_script event", () => {
  const scripts = allChatScripts();

  it("found chat scripts to test", () => {
    expect(scripts.length).toBeGreaterThan(0);
  });

  for (const [i, { scanId, events }] of scripts.entries()) {
    it(`round-trips ${scanId} turn ${i} (${events.length} events)`, () => {
      const wire = encodeWire(events);
      const parsed = parseChatStream(wire);
      // Every event parses back, in order, with keep-alives dropped.
      expect(parsed.length).toBe(events.length);

      for (let j = 0; j < events.length; j++) {
        const src = events[j];
        const got = parsed[j] as unknown as Record<string, unknown>;
        expect(got.event).toBe(src.event);
        switch (src.event) {
          case "token":
            expect(got.text).toBe(src.text);
            break;
          case "tool_call":
            expect(got.name).toBe(src.name);
            expect(got.args).toEqual(src.args);
            break;
          case "highlight":
            expect(got.part_ids).toEqual(src.part_ids);
            break;
          case "card":
            // The decoded event nests the wire {type,payload} under `card`.
            expect(got.card).toEqual(src.card);
            break;
          case "done":
            expect(got.message_id).toBe(src.message_id);
            break;
        }
      }
    });
  }

  it("ignores keep-alive comment lines and blank frames", () => {
    const wire = [
      ": keep-alive",
      "",
      "event: token",
      'data: {"text":"hi"}',
      "",
      ": keep-alive",
      "",
    ].join("\n");
    const parsed = parseChatStream(wire);
    expect(parsed).toEqual([{ event: "token", text: "hi" }]);
  });

  it("decodeChatEvent wraps a card payload under `card`", () => {
    const ev = decodeChatEvent("card", { type: "parts_to_buy", payload: [] });
    expect(ev).toEqual({ event: "card", card: { type: "parts_to_buy", payload: [] } });
  });
});
