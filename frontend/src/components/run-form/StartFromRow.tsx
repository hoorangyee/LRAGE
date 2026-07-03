import { useState } from "react"
import { FilePlus2 } from "lucide-react"

import { api } from "@/api/client"
import { useRuns } from "@/api/queries"
import type { RunDetail, RunListItem } from "@/api/types"
import { cn } from "@/lib/utils"

const MAX_CARDS = 4

interface StartFromRowProps {
  onPick: (run: RunDetail) => void
  onBlank: () => void
  startedFrom: string | null
}

export function StartFromRow({ onPick, onBlank, startedFrom }: StartFromRowProps) {
  const runs = useRuns()
  const [loadingId, setLoadingId] = useState<string | null>(null)

  const recent = (runs.data ?? [])
    .filter((r) => r.status === "completed")
    .slice(0, MAX_CARDS)

  if (recent.length === 0) return null

  const pick = async (run: RunListItem) => {
    setLoadingId(run.run_id)
    try {
      const detail = await api.get<RunDetail>(`/api/runs/${run.run_id}`)
      onPick(detail)
    } finally {
      setLoadingId(null)
    }
  }

  return (
    <div className="space-y-1.5">
      <span className="section-label">Start from</span>
      <div className="flex flex-wrap gap-2">
        {recent.map((run) => (
          <button
            key={run.run_id}
            type="button"
            onClick={() => pick(run)}
            disabled={loadingId !== null}
            className={cn(
              "w-52 rounded-lg border bg-card px-3 py-2 text-left transition-colors hover:border-muted-foreground/40",
              startedFrom === run.run_id && "border-brass ring-1 ring-brass/40",
              loadingId === run.run_id && "opacity-60"
            )}
          >
            <p className="truncate text-xs font-medium">
              {run.name || "Untitled run"}
            </p>
            <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
              {run.model}
              {run.tasks.length ? ` · ${run.tasks[0]}` : ""}
              {run.tasks.length > 1 ? ` +${run.tasks.length - 1}` : ""}
            </p>
            <p className="mt-1 font-mono text-[11px]">
              {run.headline_metric ? (
                <>
                  <span className="text-muted-foreground">
                    {run.headline_metric.metric}{" "}
                  </span>
                  <span className="text-brass">
                    {run.headline_metric.value.toFixed(4)}
                  </span>
                </>
              ) : (
                <span className="text-muted-foreground">no metrics</span>
              )}
            </p>
          </button>
        ))}
        <button
          type="button"
          onClick={onBlank}
          className={cn(
            "flex w-32 flex-col items-center justify-center gap-1 rounded-lg border border-dashed bg-transparent px-3 py-2 text-muted-foreground transition-colors hover:border-muted-foreground/50 hover:text-foreground"
          )}
        >
          <FilePlus2 className="size-4" />
          <span className="text-xs">Blank config</span>
        </button>
      </div>
    </div>
  )
}
