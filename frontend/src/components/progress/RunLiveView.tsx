import type { RunDetail } from "@/api/types"
import type { RunLogLine } from "@/api/stream"
import { Progress } from "@/components/ui/progress"
import { LogTail } from "./LogTail"
import { PhaseStepper } from "./PhaseStepper"

interface RunLiveViewProps {
  run: RunDetail
  logs: RunLogLine[]
  connected: boolean
}

export function RunLiveView({ run, logs, connected }: RunLiveViewProps) {
  const progress = run.progress

  return (
    <div className="space-y-4">
      {run.status === "queued" && (
        <p className="text-[13px] text-muted-foreground">
          Waiting in queue
          {run.queue_position != null ? ` — position ${run.queue_position}` : ""}.
          Another run is using the GPU.
        </p>
      )}

      <PhaseStepper config={run.config} phase={progress?.phase} />

      {progress?.total != null && (
        <div className="space-y-1.5">
          <div className="flex items-baseline justify-between">
            <span className="text-xs text-muted-foreground">
              {progress.desc || "Progress"}
            </span>
            <span className="data-mono text-xs text-muted-foreground">
              {progress.n}/{progress.total}
              {progress.pct != null && (
                <span className="ml-2 text-brass">{progress.pct.toFixed(0)}%</span>
              )}
            </span>
          </div>
          <Progress value={progress.pct ?? 0} className="h-1.5" />
        </div>
      )}
      {progress?.total == null && progress?.n != null && (
        <p className="data-mono text-xs text-muted-foreground">
          {progress.desc || "Processed"}: {progress.n}
        </p>
      )}

      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="section-label">Log</span>
          {!connected && run.status !== "queued" && (
            <span className="text-[11px] text-status-warn">
              stream reconnecting — falling back to polling
            </span>
          )}
        </div>
        <LogTail lines={logs} />
      </div>
    </div>
  )
}
