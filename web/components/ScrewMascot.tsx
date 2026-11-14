"use client";

import { useState } from "react";

// A stable hit area lets the screw rise without moving away from the pointer.
export function ScrewMascot() {
  const [replay, setReplay] = useState(0);
  const [entering, setEntering] = useState(true);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [tapped, setTapped] = useState(false);
  const raised = hovered || focused || tapped;

  return (
    <div className="mascot-wrap">
      <button
        type="button"
        className={`mascot-scene ${entering ? "is-entering" : ""} ${raised ? "is-raised" : ""}`}
        aria-label="Say hello to the screw"
        aria-pressed={raised}
        onPointerEnter={(e) => {
          if (e.pointerType === "mouse" || e.pointerType === "pen") {
            setEntering(false);
            setHovered(true);
          }
        }}
        onPointerLeave={() => setHovered(false)}
        onFocus={(e) => {
          if (e.currentTarget.matches(":focus-visible")) {
            setEntering(false);
            setFocused(true);
          }
        }}
        onBlur={() => { setFocused(false); setTapped(false); }}
        onClick={(e) => {
          if (e.detail > 0 && window.matchMedia("(hover: none)").matches) {
            setEntering(false);
            setTapped((value) => !value);
          }
        }}
      >
        <span className="mascot-hello" aria-hidden="true">hello!</span>
        <span className="mascot-shadow" />
        <span className="mascot-clip" aria-hidden="true">
          <span key={replay} className="mascot-driver" onAnimationEnd={(e) => {
            if (e.animationName === "screw-drive") setEntering(false);
          }}>
            <span className="mascot-bob">
              <span className="mascot-screw">
                <span className="mascot-head"><span className="mascot-eyes" /><span className="mascot-smile" /></span>
                <span className="mascot-stem" />
              </span>
            </span>
          </span>
        </span>
        <span className="mascot-surface" />
        <span className="mascot-tick mascot-tick-left" />
        <span className="mascot-tick mascot-tick-right" />
      </button>
      <button type="button" onClick={() => {
        setHovered(false); setFocused(false); setTapped(false);
        setEntering(true); setReplay((n) => n + 1);
      }} className="mascot-replay" aria-label="Replay screw entrance animation">↻ Again</button>
    </div>
  );
}
