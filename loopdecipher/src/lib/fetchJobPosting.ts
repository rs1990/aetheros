const FETCH_TIMEOUT_MS = 10000;
const MAX_CHARS = 20000;

function htmlToText(html: string): string {
  const withoutNoise = html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ");

  const withBreaks = withoutNoise.replace(/<(br|\/p|\/div|\/li|\/h[1-6])\s*\/?>/gi, "\n");

  const stripped = withBreaks
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&#39;/gi, "'")
    .replace(/&quot;/gi, '"');

  return stripped
    .split("\n")
    .map((line) => line.replace(/[ \t]+/g, " ").trim())
    .filter(Boolean)
    .join("\n");
}

export async function fetchJobDescriptionFromUrl(url: string): Promise<string> {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error("That doesn't look like a valid URL.");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("Only http/https URLs are supported.");
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(parsed.toString(), {
      signal: controller.signal,
      headers: {
        "User-Agent": "IntervueBot/1.0 (+job description fetch for interview prep tool)",
        Accept: "text/html",
      },
      redirect: "follow",
    });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error("Timed out fetching that URL. Paste the job description text instead.");
    }
    throw new Error("Couldn't reach that URL. Paste the job description text instead.");
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    throw new Error(
      `That site returned ${response.status} — it may block automated requests. Paste the job description text instead.`,
    );
  }

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("text/html") && !contentType.includes("text/plain")) {
    throw new Error("That URL didn't return a readable page. Paste the job description text instead.");
  }

  const html = await response.text();
  const text = htmlToText(html);

  if (text.length < 100) {
    throw new Error(
      "Couldn't extract meaningful text from that page — it may render its content with JavaScript. Paste the job description text instead.",
    );
  }

  return text.slice(0, MAX_CHARS);
}
