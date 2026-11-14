"use client";

import { CadResultCard, PartsToBuyCard, RepairPlanCard } from "./Cards";

type Card = { type: string; payload: unknown };

// Renders the rich chat cards for one message. Loaded lazily by ChatPanel, so
// the card code only ships once a message actually carries a card.
export function ChatCards({
  cards,
  onPartChip,
  onShowCad,
}: {
  cards: Card[];
  onPartChip: (id: string) => void;
  onShowCad: (cad: Record<string, unknown>) => void;
}) {
  return (
    <>
      {cards.map((card, i) => (
        <div key={i} className="mt-2 text-left">
          {card.type === "repair_plan" ? (
            <RepairPlanCard
              payload={card.payload as Record<string, unknown>}
              onPartChip={onPartChip}
            />
          ) : card.type === "parts_to_buy" ? (
            <PartsToBuyCard payload={card.payload as Record<string, unknown>[]} />
          ) : card.type === "cad_result" ? (
            <CadResultCard
              payload={card.payload as Record<string, unknown>}
              onShowInViewer={onShowCad}
            />
          ) : null}
        </div>
      ))}
    </>
  );
}
