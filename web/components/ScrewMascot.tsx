"use client";

import { useState } from "react";

// CSS-only entrance keeps the homepage free of the workbench's 3D dependencies.
export function ScrewMascot() {
  const [replay, setReplay] = useState(0);
  return (
    <div className="mascot-wrap">
      <div className="mascot-scene" role="img" aria-label="A friendly screw twists into the page and peeks out">
        <div className="mascot-shadow" />
        <div className="mascot-clip">
          <div key={replay} className="mascot-driver">
            <div className="mascot-screw">
              <div className="mascot-head"><span className="mascot-eyes" /><span className="mascot-smile" /></div>
              <div className="mascot-stem" />
            </div>
          </div>
        </div>
        <div className="mascot-surface" />
        <span className="mascot-tick mascot-tick-left" />
        <span className="mascot-tick mascot-tick-right" />
      </div>
      <button type="button" onClick={() => setReplay((n) => n + 1)} className="mascot-replay" aria-label="Replay screw entrance animation">↻ Again</button>
    </div>
  );
}
