export function isMockMode(): boolean {
  return (
    process.env.NEXT_PUBLIC_USE_MOCK_MODE === "true" || !process.env.ANTHROPIC_API_KEY
  );
}
