"use client";

// Render text with [[part:pX]] chips as clickable buttons. Kept in its own tiny
// module (no card/markdown deps) so the chat shell can use it without pulling
// the rich-card bundle.
export function PartText({
  text,
  onPartChip,
}: {
  text: string;
  onPartChip: (id: string) => void;
}) {
  const parts = text.split(/(\[\[part:[^\]]+\]\])/g);
  return (
    <>
      {parts.map((seg, i) => {
        const m = seg.match(/^\[\[part:([^\]]+)\]\]$/);
        if (m) {
          return (
            <button
              key={i}
              onClick={() => onPartChip(m[1])}
              className="mx-0.5 rounded bg-sky-800/70 px-1 text-sky-200 hover:bg-sky-700"
            >
              {m[1]}
            </button>
          );
        }
        return <span key={i}>{seg}</span>;
      })}
    </>
  );
}
