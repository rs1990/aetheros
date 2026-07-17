import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Intervue — Real-World Interview Prep",
  description:
    "Turn a job posting into a personalized, forum-verified interview study guide.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
