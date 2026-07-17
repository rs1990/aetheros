export type Difficulty = "Easy" | "Medium" | "Hard";

export type QuestionCategory =
  | "Technical"
  | "System Design"
  | "Coding"
  | "Behavioral";

export type QuestionSource = "predicted" | "forum-sourced";

export interface Question {
  id: string;
  text: string;
  category: QuestionCategory;
  difficulty: Difficulty;
  source: QuestionSource;
  sourceDetail?: string;
  completed?: boolean;
}

export interface CultureInsight {
  insight: string;
  source: string;
  sentiment: "positive" | "negative" | "neutral";
}

export interface StudyWeek {
  week: number;
  focus: string;
  tasks: string[];
}

export interface DecipherRequest {
  jobDescription?: string;
  jobUrl?: string;
  companyName: string;
  roleName: string;
  resumeText?: string;
}

export interface DecipherResult {
  questions: Question[];
  cultureInsights: CultureInsight[];
  mustKnowTech: string[];
  studySchedule: StudyWeek[];
  mode: "live" | "mock";
}

export type ForumSourceName = "reddit" | "hackernews" | "github";

export interface ForumSnippet {
  text: string;
  source: ForumSourceName;
  url: string;
}

export interface AnswerRequest {
  questionId: string;
  questionText: string;
  category: QuestionCategory;
  difficulty: Difficulty;
  companyName?: string;
  roleName?: string;
  resumeText?: string;
}

export interface AnswerSource {
  title: string;
  url: string;
}

export interface AnswerResult {
  answer: string;
  sources: AnswerSource[];
  diagram?: string;
  model: string;
}

export interface AtsRequest {
  resumeText: string;
  jobDescription?: string;
  jobUrl?: string;
  companyName?: string;
  roleName?: string;
}

export type AtsVerdict = "strong" | "moderate" | "weak";

export interface BulletRewrite {
  original: string;
  rewritten: string;
  reason: string;
}

export interface AtsResult {
  matchScore: number;
  verdict: AtsVerdict;
  matchedKeywords: string[];
  missingKeywords: string[];
  formattingWarnings: string[];
  bulletRewrites: BulletRewrite[];
  summary: string;
  mode: "live" | "mock";
}
