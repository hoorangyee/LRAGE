// simple_parse_args_string-compatible `k=v,k=v` helpers.

export interface ArgPair {
  key: string
  value: string
}

export function parseArgsString(args: string): ArgPair[] | null {
  const trimmed = args.trim()
  if (!trimmed) return []
  const pairs: ArgPair[] = []
  for (const part of trimmed.split(",")) {
    const eq = part.indexOf("=")
    if (eq <= 0) return null
    const key = part.slice(0, eq).trim()
    const value = part.slice(eq + 1).trim()
    if (!key || value.includes("=")) return null
    pairs.push({ key, value })
  }
  return pairs
}

export function formatArgsString(pairs: ArgPair[]): string {
  return pairs.map(({ key, value }) => `${key}=${value}`).join(",")
}

export function isValidArgsString(args: string): boolean {
  return parseArgsString(args) !== null
}
