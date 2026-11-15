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
          // Chips can sit in any inline text, including bold ("**[[part:p1]] Headband**"),
          // headings and table cells, not only directly in a paragraph.
          p: ({ children }) => <p>{renderChildren(children, onPartChip)}</p>,
          li: ({ children }) => <li>{renderChildren(children, onPartChip)}</li>,
          strong: ({ children }) => <strong>{renderChildren(children, onPartChip)}</strong>,
          em: ({ children }) => <em>{renderChildren(children, onPartChip)}</em>,
          h1: ({ children }) => <h1>{renderChildren(children, onPartChip)}</h1>,
          h2: ({ children }) => <h2>{renderChildren(children, onPartChip)}</h2>,
          h3: ({ children }) => <h3>{renderChildren(children, onPartChip)}</h3>,
          h4: ({ children }) => <h4>{renderChildren(children, onPartChip)}</h4>,
          td: ({ children }) => <td>{renderChildren(children, onPartChip)}</td>,
          th: ({ children }) => <th>{renderChildren(children, onPartChip)}</th>,
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
