/**
 * Wikipedia REST API client.
 * Pulls the page summary and extract for a landmark — sourced facts only,
 * not LLM-generated content. Every fact in the narration pipeline traces back here.
 */

const WIKI_API = 'https://en.wikipedia.org/w/api.php';
const WIKI_REST = 'https://en.wikipedia.org/api/rest_v1';

export async function fetchPageSummary(title) {
  const encoded = encodeURIComponent(title.replace(/ /g, '_'));
  const url = `${WIKI_REST}/page/summary/${encoded}`;

  const res = await fetch(url, {
    headers: { 'User-Agent': 'WayTale/1.0 (waytale-app)' },
  });

  if (!res.ok) {
    if (res.status === 404) return null;
    throw new Error(`Wikipedia summary error: ${res.status} for "${title}"`);
  }

  const data = await res.json();
  return {
    title: data.title,
    description: data.description,
    extract: data.extract,          // plain text extract (≤1000 chars usually)
    extract_html: data.extract_html,
    source_url: data.content_urls?.desktop?.page ?? `https://en.wikipedia.org/wiki/${encoded}`,
    thumbnail: data.thumbnail?.source ?? null,
    wikidata_id: data.wikibase_item,
  };
}

export async function fetchPageSections(title) {
  const params = new URLSearchParams({
    action: 'query',
    titles: title,
    prop: 'extracts|categories|links',
    exsentences: '20',       // up to 20 sentences
    exsectionformat: 'plain',
    explaintext: '1',
    cllimit: '20',
    format: 'json',
    origin: '*',
  });

  const res = await fetch(`${WIKI_API}?${params}`, {
    headers: { 'User-Agent': 'WayTale/1.0 (waytale-app)' },
  });

  if (!res.ok) throw new Error(`Wikipedia sections error: ${res.status}`);

  const data = await res.json();
  const pages = data.query?.pages ?? {};
  const page = Object.values(pages)[0];

  if (!page || page.missing !== undefined) return null;

  const categories = (page.categories ?? [])
    .map((c) => c.title.replace('Category:', ''))
    .filter((c) => !c.startsWith('Articles') && !c.startsWith('CS1') && !c.startsWith('Use'));

  return {
    title: page.title,
    extract: page.extract ?? '',
    categories,
    source_url: `https://en.wikipedia.org/wiki/${encodeURIComponent(page.title.replace(/ /g, '_'))}`,
  };
}

/**
 * Detect legend/folklore content by scanning for common markers in the text.
 * Returns 'legend' if legends detected, 'verified' otherwise.
 */
export function detectFactType(text) {
  const legendMarkers = [
    /\blegend\b/i, /\baccording to (legend|folklore|tradition|myth)\b/i,
    /\bsaid to (have|be)\b/i, /\bbelieved to\b/i, /\bpurportedly\b/i,
    /\bfolk tale\b/i, /\bmythology\b/i, /\bunverified\b/i,
  ];
  const hasLegend = legendMarkers.some((rx) => rx.test(text));
  return hasLegend ? 'mixed' : 'verified';
}
