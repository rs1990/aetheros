"use client";

import { useEffect, useRef, useState } from "react";

let diagramCounter = 0;

export default function MermaidDiagram({ definition }: { definition: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function render() {
      try {
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "strict" });
        const id = `mermaid-diagram-${diagramCounter++}`;
        const { svg } = await mermaid.render(id, definition);
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      } catch {
        if (!cancelled) setError("Couldn't render this diagram.");
      }
    }

    render();
    return () => {
      cancelled = true;
    };
  }, [definition]);

  if (error) {
    return (
      <pre className="whitespace-pre-wrap rounded-lg border border-slate-800 bg-slate-950 p-3 text-xs text-slate-400">
        {definition}
      </pre>
    );
  }

  return <div ref={containerRef} className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-950 p-3" />;
}
