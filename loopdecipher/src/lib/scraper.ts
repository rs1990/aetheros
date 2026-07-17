import type { ForumSnippet } from "./types";

const REDDIT_SUBREDDITS = ["leetcode", "ExperiencedDevs", "cscareerquestions", "csMajors"];
const HN_ENDPOINT = "https://hn.algolia.com/api/v1/search";
const GITHUB_SEARCH_ENDPOINT = "https://api.github.com/search/code";

let redditToken: { value: string; expiresAt: number } | null = null;

async function getRedditToken(): Promise<string | null> {
  const clientId = process.env.REDDIT_CLIENT_ID;
  const clientSecret = process.env.REDDIT_CLIENT_SECRET;
  if (!clientId || !clientSecret) return null;

  if (redditToken && redditToken.expiresAt > Date.now()) {
    return redditToken.value;
  }

  const res = await fetch("https://www.reddit.com/api/v1/access_token", {
    method: "POST",
    headers: {
      Authorization: `Basic ${Buffer.from(`${clientId}:${clientSecret}`).toString("base64")}`,
      "Content-Type": "application/x-www-form-urlencoded",
      "User-Agent": "LoopDecipher/1.0 (interview-prep-tool)",
    },
    body: "grant_type=client_credentials",
  });
  if (!res.ok) return null;

  const data = (await res.json()) as { access_token: string; expires_in: number };
  redditToken = { value: data.access_token, expiresAt: Date.now() + data.expires_in * 1000 };
  return redditToken.value;
}

async function fetchRedditThreads(company: string, role: string): Promise<ForumSnippet[]> {
  const token = await getRedditToken();
  if (!token) return [];

  const query = encodeURIComponent(`${company} ${role} interview questions`);
  const snippets: ForumSnippet[] = [];

  for (const subreddit of REDDIT_SUBREDDITS) {
    const res = await fetch(
      `https://oauth.reddit.com/r/${subreddit}/search?q=${query}&restrict_sr=1&sort=relevance&limit=5`,
      {
        headers: {
          Authorization: `Bearer ${token}`,
          "User-Agent": "LoopDecipher/1.0 (interview-prep-tool)",
        },
      },
    );
    if (!res.ok) continue;

    const data = (await res.json()) as {
      data: { children: { data: { title: string; selftext: string; permalink: string } }[] };
    };
    for (const child of data.data.children) {
      const text = `${child.data.title}\n${child.data.selftext}`.trim().slice(0, 1500);
      if (text.length > 20) {
        snippets.push({
          text,
          source: "reddit",
          url: `https://reddit.com${child.data.permalink}`,
        });
      }
    }
  }

  return snippets;
}

async function fetchHackerNews(company: string, role: string): Promise<ForumSnippet[]> {
  const query = `${company} ${role} interview`;
  const res = await fetch(
    `${HN_ENDPOINT}?query=${encodeURIComponent(query)}&tags=comment&hitsPerPage=10`,
    { headers: { "User-Agent": "LoopDecipher/1.0 (interview-prep-tool)" } },
  );
  if (!res.ok) return [];

  const data = (await res.json()) as {
    hits: { comment_text: string | null; objectID: string }[];
  };

  return data.hits
    .filter((hit) => hit.comment_text && hit.comment_text.length > 40)
    .map((hit) => ({
      text: hit.comment_text!.replace(/<[^>]+>/g, "").slice(0, 1500),
      source: "hackernews" as const,
      url: `https://news.ycombinator.com/item?id=${hit.objectID}`,
    }));
}

async function fetchGitHubCompanyQuestions(company: string): Promise<ForumSnippet[]> {
  const token = process.env.GITHUB_TOKEN;
  const headers: Record<string, string> = {
    Accept: "application/vnd.github+json",
    "User-Agent": "LoopDecipher/1.0 (interview-prep-tool)",
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  const query = encodeURIComponent(`${company} interview questions in:file filename:.md`);
  const res = await fetch(`${GITHUB_SEARCH_ENDPOINT}?q=${query}&per_page=5`, { headers });
  if (!res.ok) return [];

  const data = (await res.json()) as {
    items: { name: string; html_url: string; repository: { full_name: string } }[];
  };

  return data.items.map((item) => ({
    text: `Community-maintained question list: ${item.repository.full_name}/${item.name}`,
    source: "github" as const,
    url: item.html_url,
  }));
}

export async function scrapeForumSnippets(
  company: string,
  role: string,
): Promise<ForumSnippet[]> {
  const results = await Promise.allSettled([
    fetchRedditThreads(company, role),
    fetchHackerNews(company, role),
    fetchGitHubCompanyQuestions(company),
  ]);

  return results.flatMap((result) => (result.status === "fulfilled" ? result.value : []));
}
