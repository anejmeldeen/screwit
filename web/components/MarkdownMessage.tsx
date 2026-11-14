"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { PartText } from "@/components/cards/PartText";

// Assistant message body rendered as markdown with [[part:pX]] chips wired to
// the viewer. Loaded lazily by ChatPanel so react-markdown stays out of the
// workspace shell bundle.
export function MarkdownMessage({
  text,
  onPartChip,
}: {
  text: string;
  onPartChip: (id: string) => void;
}) {
  return (
    <div className="prose prose-sm max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p>{renderChildren(children, onPartChip)}</p>,
          li: ({ children }) => <li>{renderChildren(children, onPartChip)}</li>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

function renderChildren(
  children: React.ReactNode,
  onPartChip: (id: string) => void,
): React.ReactNode {
  if (typeof children === "string") {
    return <PartText text={children} onPartChip={onPartChip} />;
  }
  if (Array.isArray(children)) {
    return children.map((c, i) =>
      typeof c === "string" ? (
        <PartText key={i} text={c} onPartChip={onPartChip} />
      ) : (
        c
      ),
    );
  }
  return children;
}
