"use client";

import { useCallback, useRef, useState } from "react";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";

const FILLER_WORDS = ["um", "uh", "like", "you know", "basically", "actually", "sort of", "kind of"];
const STAR_SIGNALS: Record<string, string[]> = {
  Situation: ["situation", "context", "background", "at the time", "we were"],
  Task: ["my task", "i needed to", "the goal", "responsible for", "had to"],
  Action: ["i decided", "i built", "i implemented", "i worked", "i led", "so i"],
  Result: ["as a result", "outcome", "resulted in", "we achieved", "improved", "reduced", "%"],
};

interface AnalysisResult {
  wpm: number;
  fillerCount: number;
  fillerBreakdown: Record<string, number>;
  starCoverage: Record<string, boolean>;
  transcript: string;
}

// The Web Speech API's SpeechRecognition constructor isn't in the standard DOM lib yet.
interface SpeechRecognitionResultEvent {
  results: ArrayLike<ArrayLike<{ transcript: string }>>;
}

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionResultEvent) => void) | null;
  onerror: ((event: unknown) => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

function getRecognition(): SpeechRecognitionLike | null {
  if (typeof window === "undefined") return null;
  const win = window as typeof window & {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  const Ctor = win.SpeechRecognition || win.webkitSpeechRecognition;
  return Ctor ? new Ctor() : null;
}

function analyzeTranscript(transcript: string, durationSeconds: number): AnalysisResult {
  const words = transcript.trim().split(/\s+/).filter(Boolean);
  const wpm = durationSeconds > 0 ? Math.round((words.length / durationSeconds) * 60) : 0;

  const lower = transcript.toLowerCase();
  const fillerBreakdown: Record<string, number> = {};
  let fillerCount = 0;
  for (const filler of FILLER_WORDS) {
    const matches = lower.match(new RegExp(`\\b${filler}\\b`, "g"));
    if (matches) {
      fillerBreakdown[filler] = matches.length;
      fillerCount += matches.length;
    }
  }

  const starCoverage: Record<string, boolean> = {};
  for (const [component, signals] of Object.entries(STAR_SIGNALS)) {
    starCoverage[component] = signals.some((signal) => lower.includes(signal));
  }

  return { wpm, fillerCount, fillerBreakdown, starCoverage, transcript };
}

export default function AudioSandbox() {
  const [supported] = useState(() => getRecognition() !== null);
  const [recording, setRecording] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const transcriptRef = useRef("");
  const startTimeRef = useRef(0);

  const start = useCallback(() => {
    const recognition = getRecognition();
    if (!recognition) return;

    transcriptRef.current = "";
    startTimeRef.current = Date.now();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.onresult = (event: SpeechRecognitionResultEvent) => {
      let finalText = "";
      for (let i = 0; i < event.results.length; i++) {
        finalText += event.results[i][0].transcript + " ";
      }
      transcriptRef.current = finalText;
    };
    recognition.onerror = () => setRecording(false);

    recognition.start();
    recognitionRef.current = recognition;
    setResult(null);
    setRecording(true);
  }, []);

  const stop = useCallback(() => {
    recognitionRef.current?.stop();
    setRecording(false);
    const durationSeconds = (Date.now() - startTimeRef.current) / 1000;
    if (transcriptRef.current.trim()) {
      setResult(analyzeTranscript(transcriptRef.current, durationSeconds));
    }
  }, []);

  return (
    <Card>
      <h2 className="mb-1 text-lg font-semibold text-slate-100">Audio Sandbox</h2>
      <p className="mb-4 text-xs text-slate-500">
        Record a behavioral answer out loud. Get WPM, filler-word counts, and a rough STAR-method check.
      </p>

      {!supported ? (
        <p className="text-sm text-amber-400">
          Speech recognition isn&apos;t supported in this browser. Try Chrome or Edge.
        </p>
      ) : (
        <>
          <Button onClick={recording ? stop : start} variant={recording ? "secondary" : "primary"}>
            {recording ? "Stop Recording" : "Start Recording"}
          </Button>

          {recording && <p className="mt-3 text-sm text-indigo-400 animate-pulse">Listening...</p>}

          {result && (
            <div className="mt-5 space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-lg border border-slate-800 p-3 text-center">
                  <p className="text-2xl font-bold text-slate-100">{result.wpm}</p>
                  <p className="text-xs text-slate-500">words per minute</p>
                </div>
                <div className="rounded-lg border border-slate-800 p-3 text-center">
                  <p className="text-2xl font-bold text-slate-100">{result.fillerCount}</p>
                  <p className="text-xs text-slate-500">filler words</p>
                </div>
              </div>

              <div>
                <p className="mb-2 text-sm font-semibold text-slate-300">STAR Coverage (approximate)</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(result.starCoverage).map(([component, present]) => (
                    <span
                      key={component}
                      className={`rounded-full border px-3 py-1 text-xs ${present ? "border-emerald-800 bg-emerald-950/40 text-emerald-400" : "border-slate-700 text-slate-500"}`}
                    >
                      {component} {present ? "✓" : "—"}
                    </span>
                  ))}
                </div>
              </div>

              <details className="text-xs text-slate-500">
                <summary className="cursor-pointer text-slate-400">View transcript</summary>
                <p className="mt-2 whitespace-pre-wrap">{result.transcript}</p>
              </details>
            </div>
          )}
        </>
      )}
    </Card>
  );
}
