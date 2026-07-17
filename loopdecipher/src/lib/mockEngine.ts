import type { CultureInsight, Difficulty, Question, QuestionCategory, StudyWeek } from "./types";

interface RawQuestion {
  text: string;
  category: QuestionCategory;
  difficulty: Difficulty;
}

const BEHAVIORAL: RawQuestion[] = [
  { text: "Tell me about a time you disagreed with a teammate's technical decision. How did you handle it?", category: "Behavioral", difficulty: "Easy" },
  { text: "Describe a project that failed. What was your role and what did you learn?", category: "Behavioral", difficulty: "Easy" },
  { text: "Tell me about a time you had to deliver bad news to a stakeholder.", category: "Behavioral", difficulty: "Easy" },
  { text: "Describe a time you had to learn a new technology quickly to complete a project.", category: "Behavioral", difficulty: "Easy" },
  { text: "Tell me about a time you received critical feedback. How did you respond?", category: "Behavioral", difficulty: "Easy" },
  { text: "Describe a situation where you had to influence someone without direct authority.", category: "Behavioral", difficulty: "Medium" },
  { text: "Tell me about the most challenging bug you've debugged. Walk me through your process.", category: "Behavioral", difficulty: "Medium" },
  { text: "Describe a time you missed a deadline. What happened and what would you do differently?", category: "Behavioral", difficulty: "Medium" },
  { text: "Tell me about a time you had to make a decision with incomplete information.", category: "Behavioral", difficulty: "Medium" },
  { text: "Describe a conflict with a manager and how you resolved it.", category: "Behavioral", difficulty: "Medium" },
  { text: "Tell me about a time you mentored a junior engineer.", category: "Behavioral", difficulty: "Easy" },
  { text: "Describe a time you had to push back on a product requirement.", category: "Behavioral", difficulty: "Medium" },
  { text: "Tell me about a time you identified a problem no one else had noticed.", category: "Behavioral", difficulty: "Medium" },
  { text: "Describe how you prioritize when you have multiple competing deadlines.", category: "Behavioral", difficulty: "Easy" },
  { text: "Tell me about a time you took ownership of something outside your job description.", category: "Behavioral", difficulty: "Medium" },
  { text: "Describe a time your code caused a production incident. What happened next?", category: "Behavioral", difficulty: "Hard" },
  { text: "Tell me about a time you had to say no to your manager.", category: "Behavioral", difficulty: "Hard" },
  { text: "Describe the biggest technical risk you've taken and whether it paid off.", category: "Behavioral", difficulty: "Hard" },
  { text: "Tell me about a time you built consensus across teams with conflicting priorities.", category: "Behavioral", difficulty: "Hard" },
  { text: "Why do you want to work here, and why this role specifically?", category: "Behavioral", difficulty: "Easy" },
];

const CODING: RawQuestion[] = [
  { text: "Reverse a linked list, iteratively and recursively.", category: "Coding", difficulty: "Easy" },
  { text: "Determine if a string has all unique characters without extra data structures.", category: "Coding", difficulty: "Easy" },
  { text: "Merge two sorted arrays in place.", category: "Coding", difficulty: "Easy" },
  { text: "Find the first non-repeating character in a string.", category: "Coding", difficulty: "Easy" },
  { text: "Implement a stack using two queues.", category: "Coding", difficulty: "Easy" },
  { text: "Check if a binary tree is balanced.", category: "Coding", difficulty: "Easy" },
  { text: "Find the missing number in an array of 1 to N.", category: "Coding", difficulty: "Easy" },
  { text: "Implement binary search on a rotated sorted array.", category: "Coding", difficulty: "Medium" },
  { text: "Given a matrix, rotate it 90 degrees in place.", category: "Coding", difficulty: "Medium" },
  { text: "Find the longest substring without repeating characters.", category: "Coding", difficulty: "Medium" },
  { text: "Implement an LRU cache with O(1) get and put.", category: "Coding", difficulty: "Medium" },
  { text: "Given two strings, determine if one is a permutation of the other.", category: "Coding", difficulty: "Easy" },
  { text: "Find all anagrams of a pattern within a string.", category: "Coding", difficulty: "Medium" },
  { text: "Serialize and deserialize a binary tree.", category: "Coding", difficulty: "Medium" },
  { text: "Detect a cycle in a directed graph.", category: "Coding", difficulty: "Medium" },
  { text: "Find the kth largest element in an unsorted array.", category: "Coding", difficulty: "Medium" },
  { text: "Implement a trie and support prefix search.", category: "Coding", difficulty: "Medium" },
  { text: "Given an array of intervals, merge all overlapping intervals.", category: "Coding", difficulty: "Medium" },
  { text: "Find the number of islands in a 2D grid.", category: "Coding", difficulty: "Medium" },
  { text: "Design a data structure that supports insert, delete, and getRandom in O(1).", category: "Coding", difficulty: "Medium" },
  { text: "Given a list of dependencies, return a valid build order (topological sort).", category: "Coding", difficulty: "Hard" },
  { text: "Find the median of two sorted arrays in O(log(m+n)).", category: "Coding", difficulty: "Hard" },
  { text: "Implement a thread-safe bounded blocking queue.", category: "Coding", difficulty: "Hard" },
  { text: "Given a stream of integers, find the median at any point (running median).", category: "Coding", difficulty: "Hard" },
  { text: "Design and implement an in-memory key-value store with TTL expiration.", category: "Coding", difficulty: "Hard" },
  { text: "Find the shortest path in a weighted graph with negative edges (Bellman-Ford).", category: "Coding", difficulty: "Hard" },
  { text: "Implement regular expression matching with support for '.' and '*'.", category: "Coding", difficulty: "Hard" },
  { text: "Design a rate limiter (token bucket or sliding window).", category: "Coding", difficulty: "Hard" },
  { text: "Given a large log file, find the top-K most frequent IP addresses under memory constraints.", category: "Coding", difficulty: "Hard" },
  { text: "Implement concurrent producer-consumer with backpressure using a fixed-size buffer.", category: "Coding", difficulty: "Hard" },
];

const SYSTEM_DESIGN: RawQuestion[] = [
  { text: "Design a URL shortener.", category: "System Design", difficulty: "Easy" },
  { text: "Design a rate limiter for a public API.", category: "System Design", difficulty: "Easy" },
  { text: "Design a simple key-value store.", category: "System Design", difficulty: "Easy" },
  { text: "Design a parking garage system.", category: "System Design", difficulty: "Easy" },
  { text: "Design a notification system (push, email, SMS).", category: "System Design", difficulty: "Medium" },
  { text: "Design a distributed cache like Memcached.", category: "System Design", difficulty: "Medium" },
  { text: "Design a news feed system (like Twitter/X).", category: "System Design", difficulty: "Medium" },
  { text: "Design a ride-sharing dispatch system.", category: "System Design", difficulty: "Medium" },
  { text: "Design a scalable chat application (like WhatsApp).", category: "System Design", difficulty: "Medium" },
  { text: "Design an autocomplete / typeahead service.", category: "System Design", difficulty: "Medium" },
  { text: "Design a distributed job scheduler.", category: "System Design", difficulty: "Medium" },
  { text: "Design a video streaming platform (like YouTube).", category: "System Design", difficulty: "Hard" },
  { text: "Design a distributed unique ID generator (like Snowflake).", category: "System Design", difficulty: "Medium" },
  { text: "Design a web crawler that scales to billions of pages.", category: "System Design", difficulty: "Hard" },
  { text: "Design a payments system with idempotency and exactly-once semantics.", category: "System Design", difficulty: "Hard" },
  { text: "Design a distributed logging and metrics pipeline.", category: "System Design", difficulty: "Hard" },
  { text: "Design an in-memory database with support for transactions.", category: "System Design", difficulty: "Hard" },
  { text: "Design a multi-region deployment strategy with active-active failover.", category: "System Design", difficulty: "Hard" },
  { text: "Design a system to detect and mitigate fraud in real time.", category: "System Design", difficulty: "Hard" },
  { text: "Design a distributed lock manager.", category: "System Design", difficulty: "Hard" },
];

const TECHNICAL: RawQuestion[] = [
  { text: "Explain the difference between processes and threads.", category: "Technical", difficulty: "Easy" },
  { text: "What is the difference between SQL and NoSQL databases, and when would you choose each?", category: "Technical", difficulty: "Easy" },
  { text: "Explain how HTTPS establishes a secure connection (TLS handshake).", category: "Technical", difficulty: "Medium" },
  { text: "What is database indexing and how does a B-tree index work?", category: "Technical", difficulty: "Medium" },
  { text: "Explain the CAP theorem with a real-world example.", category: "Technical", difficulty: "Medium" },
  { text: "What is the difference between optimistic and pessimistic locking?", category: "Technical", difficulty: "Medium" },
  { text: "Explain how garbage collection works in a language of your choice.", category: "Technical", difficulty: "Medium" },
  { text: "What are the tradeoffs between REST and gRPC?", category: "Technical", difficulty: "Medium" },
  { text: "Explain eventual consistency vs. strong consistency.", category: "Technical", difficulty: "Medium" },
  { text: "What is a deadlock and how would you detect and prevent one?", category: "Technical", difficulty: "Medium" },
  { text: "Explain how a hash table resolves collisions.", category: "Technical", difficulty: "Easy" },
  { text: "What is the difference between horizontal and vertical scaling?", category: "Technical", difficulty: "Easy" },
  { text: "Explain how a load balancer decides where to route traffic.", category: "Technical", difficulty: "Easy" },
  { text: "What is idempotency and why does it matter for distributed systems?", category: "Technical", difficulty: "Medium" },
  { text: "Explain the difference between synchronous and asynchronous replication.", category: "Technical", difficulty: "Medium" },
  { text: "What is a message queue used for, and how does it differ from a pub/sub system?", category: "Technical", difficulty: "Medium" },
  { text: "Explain how DNS resolution works end to end.", category: "Technical", difficulty: "Medium" },
  { text: "What is the difference between a mutex and a semaphore?", category: "Technical", difficulty: "Medium" },
  { text: "Explain sharding strategies for a relational database.", category: "Technical", difficulty: "Hard" },
  { text: "What is Raft/Paxos consensus and why do distributed systems need it?", category: "Technical", difficulty: "Hard" },
  { text: "Explain how a JIT compiler improves runtime performance.", category: "Technical", difficulty: "Hard" },
  { text: "What is backpressure and how do streaming systems handle it?", category: "Technical", difficulty: "Hard" },
  { text: "Explain vector clocks and why they're used for causality tracking.", category: "Technical", difficulty: "Hard" },
  { text: "What is the difference between at-least-once, at-most-once, and exactly-once delivery?", category: "Technical", difficulty: "Hard" },
  { text: "Explain how a distributed hash table (DHT) like Chord works.", category: "Technical", difficulty: "Hard" },
  { text: "What are the tradeoffs of using an LSM-tree vs. a B-tree for storage engines?", category: "Technical", difficulty: "Hard" },
  { text: "Explain gossip protocols and where they're used in distributed systems.", category: "Technical", difficulty: "Hard" },
  { text: "What is a Bloom filter and when would you use one?", category: "Technical", difficulty: "Medium" },
  { text: "Explain the tradeoffs between microservices and a monolith.", category: "Technical", difficulty: "Medium" },
  { text: "How would you debug a memory leak in a long-running service?", category: "Technical", difficulty: "Hard" },
];

const RAW_QUESTIONS: RawQuestion[] = [
  ...BEHAVIORAL,
  ...CODING,
  ...SYSTEM_DESIGN,
  ...TECHNICAL,
];

export const MOCK_QUESTIONS: Question[] = RAW_QUESTIONS.map((q, index) => ({
  id: `mock-${index + 1}`,
  text: q.text,
  category: q.category,
  difficulty: q.difficulty,
  source: "predicted",
  completed: false,
}));

export const MOCK_CULTURE_INSIGHTS: CultureInsight[] = [
  {
    insight: "Interview loops tend to move fast — expect a decision within one to two weeks of the onsite.",
    source: "General industry pattern",
    sentiment: "positive",
  },
  {
    insight: "System design rounds often weight communication and tradeoff articulation as heavily as the final design.",
    source: "General industry pattern",
    sentiment: "neutral",
  },
  {
    insight: "Take-home assignments, when present, are usually scoped for 2-4 hours — budget time accordingly.",
    source: "General industry pattern",
    sentiment: "neutral",
  },
];

export const MOCK_STUDY_SCHEDULE: StudyWeek[] = [
  { week: 1, focus: "Foundations & Easy Problems", tasks: ["Review core data structures", "Solve 10 Easy coding questions", "Draft your 2-3 behavioral stories"] },
  { week: 2, focus: "Core Algorithms", tasks: ["Solve 10 Medium coding questions", "Practice explaining time/space complexity out loud", "Review one system design primer chapter"] },
  { week: 3, focus: "System Design & Depth", tasks: ["Work through 3 system design prompts end-to-end", "Solve 5 Hard coding questions", "Record yourself answering 2 behavioral questions"] },
  { week: 4, focus: "Mock Interviews", tasks: ["Run 2 full mock interviews (coding + behavioral)", "Review weak areas from mocks", "Refine your resume walkthrough to under 90 seconds"] },
  { week: 5, focus: "Polish & Company Research", tasks: ["Research the company's recent product launches", "Prepare 5 thoughtful questions for the interviewer", "Do a final pass on your top 10 weakest topics"] },
];

export function buildMockResult(mustKnowTech: string[]) {
  return {
    questions: MOCK_QUESTIONS,
    cultureInsights: MOCK_CULTURE_INSIGHTS,
    mustKnowTech,
    studySchedule: MOCK_STUDY_SCHEDULE,
    mode: "mock" as const,
  };
}
