export async function parseResumeFile(file: File): Promise<string> {
  const name = file.name.toLowerCase();

  if (name.endsWith(".txt") || file.type.startsWith("text/")) {
    return file.text();
  }

  if (name.endsWith(".pdf") || name.endsWith(".docx") || name.endsWith(".doc")) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/api/parse-resume", { method: "POST", body: formData });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(body.error || "Failed to parse resume file.");
    }
    return body.text as string;
  }

  throw new Error("Unsupported file type — upload a .txt, .pdf, or .docx file.");
}
