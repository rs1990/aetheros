import type { HTMLAttributes } from "react";

export default function Card({ className = "", ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`rounded-xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg ${className}`}
      {...props}
    />
  );
}
