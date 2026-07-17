import { NextResponse } from "next/server";
import { PDFParse } from "pdf-parse";
import mammoth from "mammoth";

const MAX_BYTES = 10 * 1024 * 1024;

export async function POST(request: Request) {
  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json({ error: "No file uploaded." }, { status: 400 });
  }
  const file = formData.get("file");

  if (!file || !(file instanceof File)) {
    return NextResponse.json({ error: "No file uploaded." }, { status: 400 });
  }
  if (file.size > MAX_BYTES) {
    return NextResponse.json({ error: "File is too large (max 10MB)." }, { status: 413 });
  }

  const name = file.name.toLowerCase();
  const buffer = Buffer.from(await file.arrayBuffer());

  try {
    if (name.endsWith(".pdf")) {
      const parser = new PDFParse({ data: new Uint8Array(buffer) });
      const result = await parser.getText();
      return NextResponse.json({ text: result.text.trim() });
    }

    if (name.endsWith(".docx")) {
      const result = await mammoth.extractRawText({ buffer });
      return NextResponse.json({ text: result.value.trim() });
    }

    if (name.endsWith(".doc")) {
      return NextResponse.json(
        { error: "Legacy .doc files aren't supported — save as .docx or .pdf and re-upload." },
        { status: 415 },
      );
    }

    return NextResponse.json(
      { error: "Unsupported file type — upload a .txt, .pdf, or .docx file." },
      { status: 415 },
    );
  } catch (error) {
    console.error("Resume parsing failed:", error);
    return NextResponse.json(
      { error: "Couldn't extract text from that file — it may be corrupted, scanned/image-only, or encrypted." },
      { status: 422 },
    );
  }
}
