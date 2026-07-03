import { useMemo, useState } from "react"

export interface SortState {
  key: string
  direction: "asc" | "desc"
}

export function useSortableColumns<T>(
  rows: T[],
  accessors: Record<string, (row: T) => string | number | null>
) {
  const [sort, setSort] = useState<SortState | null>(null)

  const sorted = useMemo(() => {
    if (!sort) return rows
    const accessor = accessors[sort.key]
    if (!accessor) return rows
    const factor = sort.direction === "asc" ? 1 : -1
    return [...rows].sort((a, b) => {
      const va = accessor(a)
      const vb = accessor(b)
      if (va == null) return 1
      if (vb == null) return -1
      if (typeof va === "number" && typeof vb === "number")
        return (va - vb) * factor
      return String(va).localeCompare(String(vb)) * factor
    })
  }, [rows, sort, accessors])

  const toggleSort = (key: string) => {
    setSort((prev) =>
      prev?.key === key
        ? prev.direction === "asc"
          ? { key, direction: "desc" }
          : null
        : { key, direction: "asc" }
    )
  }

  return { sorted, sort, toggleSort }
}
