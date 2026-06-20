import { useState } from "react";
import { Daily } from "./pages/Daily";
import { Weekly } from "./pages/Weekly";
import { Inventory } from "./pages/Inventory";
import { clsx } from "clsx";

type Page = "daily" | "weekly" | "inventory";

const NAV: { id: Page; label: string }[] = [
  { id: "daily", label: "Daily Ops" },
  { id: "weekly", label: "Weekly Intel" },
  { id: "inventory", label: "Inventory" },
];

export default function App() {
  const [page, setPage] = useState<Page>("daily");

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Sidebar */}
      <div className="flex h-screen">
        <aside className="w-48 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
          <div className="p-4 border-b border-gray-800">
            <div className="text-white font-bold text-sm">Supply Chain</div>
            <div className="text-gray-500 text-xs">Intelligence Platform</div>
          </div>
          <nav className="p-2 flex-1">
            {NAV.map(n => (
              <button
                key={n.id}
                onClick={() => setPage(n.id)}
                className={clsx(
                  "w-full text-left px-3 py-2 rounded-lg text-sm mb-1 transition-colors",
                  page === n.id
                    ? "bg-blue-900 text-blue-100 font-medium"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
                )}
              >
                {n.label}
              </button>
            ))}
          </nav>
          <div className="p-4 border-t border-gray-800">
            <div className="text-gray-600 text-xs">PACCAR Supply Chain</div>
            <div className="text-gray-700 text-xs">v1.0.0</div>
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 overflow-y-auto p-6">
          {page === "daily" && <Daily />}
          {page === "weekly" && <Weekly />}
          {page === "inventory" && <Inventory />}
        </main>
      </div>
    </div>
  );
}
