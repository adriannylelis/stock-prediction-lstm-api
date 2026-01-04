export function validateTicker(ticker: string): boolean {
  const s = ticker.trim().toUpperCase();
  return /^[A-Z]{1,5}$/.test(s);
}

export function normalizeTicker(ticker: string): string {
  return ticker.trim().toUpperCase();
}
