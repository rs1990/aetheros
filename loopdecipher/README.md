# LoopDecipher

Turn a job posting into a personalized interview study guide. LoopDecipher pulls a job description
apart to find the must-know technologies, cross-references real interview discussion from public,
API-accessible sources (Reddit, Hacker News, and community-maintained GitHub question banks), and
uses Claude to merge it all into a ~100-question study plan — separating **forum-verified**
questions from **predicted** ones.

## Why no Glassdoor/Teamblind scraping

Those sites actively block automated scraping with anti-bot protection. Circumventing that would
mean impersonating a browser to defeat access controls the site operator put there on purpose —
that's a ToS/authorization problem, not just a scraping one, so this project doesn't do it.
Instead, forum signal comes from sources with public, terms-compliant APIs: Reddit's official API,
Hacker News' Algolia search API, and GitHub's code search API.

## Stack

- **Frontend:** Next.js (App Router), React, Tailwind CSS
- **Backend:** Next.js Route Handlers (TypeScript)
- **AI:** Anthropic SDK (`claude-opus-4-8` by default, configurable)
- **Data sources:** Reddit API (OAuth client-credentials), Hacker News Algolia API, GitHub Search API
- **Storage:** Local Storage on the client; no database required

## Quickstart

```bash
npm install
cp .env.example .env.local   # fill in keys, or leave blank for mock mode
npm run dev
```

Open http://localhost:3000.

### Modes

- **Mock mode** (default with no `ANTHROPIC_API_KEY`): serves the built-in 100-question baseline
  bank, keyword-matched against the job description. No network calls, no API keys needed.
- **Live mode** (`ANTHROPIC_API_KEY` set): scrapes Reddit/HN/GitHub for real discussion about the
  company + role, sends it to Claude alongside the job description and optional resume text, and
  returns a synthesized guide with forum-verified questions flagged. Reddit and GitHub credentials
  are optional — the app degrades gracefully to whichever sources have credentials configured.

## Getting API credentials

| Service | Where | Cost |
|---|---|---|
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | Pay-per-token |
| Reddit (script app) | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) | Free |
| GitHub token (optional) | [github.com/settings/tokens](https://github.com/settings/tokens) | Free |

## Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/loopdecipher&env=ANTHROPIC_API_KEY,REDDIT_CLIENT_ID,REDDIT_CLIENT_SECRET,GITHUB_TOKEN&envDescription=API%20keys%20used%20for%20live%20synthesis%20and%20forum%20scraping.%20Leave%20blank%20for%20mock%20mode.)

## Architecture

```
loopdecipher/
├── src/
│   ├── app/
│   │   ├── layout.tsx, page.tsx      Dashboard shell
│   │   └── api/decipher/route.ts     Dual-mode synthesis endpoint
│   ├── components/                   InputForm, QuestionBank, CultureDecoder,
│   │                                 PitchGenerator, StudySchedule, AudioSandbox, LoopTimeline
│   └── lib/
│       ├── scraper.ts                Reddit/HN/GitHub API clients
│       ├── mockEngine.ts             100-question offline baseline
│       └── types.ts                  Shared TypeScript interfaces
```
