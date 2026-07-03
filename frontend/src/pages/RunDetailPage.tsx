import { Copy } from "lucide-react"
import {
  Link,
  NavLink,
  Navigate,
  Route,
  Routes,
  useParams,
} from "react-router-dom"

import { useResults, useRun, useRunLogs } from "@/api/queries"
import { useRunStream } from "@/api/stream"
import type { RunDetail } from "@/api/types"
import { PageHeader } from "@/components/layout/PageHeader"
import { CancelRunButton } from "@/components/progress/CancelRunButton"
import { RunLiveView } from "@/components/progress/RunLiveView"
import { MetricsTable } from "@/components/results/MetricsTable"
import { SamplesBrowser } from "@/components/results/SamplesBrowser"
import { ConfigView } from "@/components/runs/ConfigView"
import { DeleteRunButton } from "@/components/runs/DeleteRunButton"
import { RunStatusBadge } from "@/components/runs/RunStatusBadge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"

const ACTIVE = new Set(["queued", "running", "cancelling"])

const TABS = [
  { path: "", label: "Overview", end: true },
  { path: "/samples", label: "Samples", end: false },
  { path: "/config", label: "Config", end: false },
  { path: "/logs", label: "Logs", end: false },
]

export function RunDetailPage() {
  const { runId } = useParams()
  const run = useRun(runId, { poll: true })
  const isActive = !!run.data && ACTIVE.has(run.data.status)
  const stream = useRunStream(runId, isActive)

  if (run.isLoading) {
    return (
      <div className="p-6">
        <Skeleton className="h-24 w-full" />
      </div>
    )
  }
  if (run.isError || !run.data) {
    return <div className="p-6 text-[13px] text-status-err">Run not found.</div>
  }

  const data = run.data
  return (
    <div>
      <PageHeader
        title={data.name || "Untitled run"}
        description={data.run_id}
        actions={
          <div className="flex items-center gap-2">
            {!isActive && (
              <>
                <DeleteRunButton runId={data.run_id} />
                <Button asChild variant="outline" size="sm" className="gap-1.5">
                  <Link to={`/new?from=${data.run_id}`}>
                    <Copy className="size-3" /> Duplicate
                  </Link>
                </Button>
              </>
            )}
            {isActive && (
              <CancelRunButton runId={data.run_id} status={data.status} />
            )}
            <RunStatusBadge status={data.status} />
          </div>
        }
      />

      {isActive ? (
        <div className="p-6">
          <RunLiveView run={data} logs={stream.logs} connected={stream.connected} />
        </div>
      ) : (
        <>
          <nav className="flex gap-1 border-b px-6">
            {TABS.map((tab) => (
              <NavLink
                key={tab.path}
                to={`/runs/${data.run_id}${tab.path}`}
                end={tab.end}
                className={({ isActive: active }) =>
                  cn(
                    "-mb-px border-b-2 px-3 py-2 text-[13px] transition-colors",
                    active
                      ? "border-brass font-medium text-foreground"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                  )
                }
              >
                {tab.label}
              </NavLink>
            ))}
          </nav>
          <div className="p-6">
            <Routes>
              <Route index element={<OverviewTab run={data} />} />
              <Route
                path="samples"
                element={<SamplesBrowser runId={data.run_id} />}
              />
              <Route path="config" element={<ConfigView config={data.config} />} />
              <Route path="logs" element={<LogsTab runId={data.run_id} />} />
              <Route path="*" element={<Navigate to="." replace />} />
            </Routes>
          </div>
        </>
      )}
    </div>
  )
}

function OverviewTab({ run }: { run: RunDetail }) {
  const results = useResults(run.run_id, run.status === "completed")

  if (run.status !== "completed") {
    return (
      <div className="space-y-4">
        {run.error && (
          <div className="space-y-1.5">
            <span className="section-label">Error</span>
            <pre className="overflow-x-auto rounded-lg border border-status-err/30 bg-card p-3 font-mono text-xs leading-relaxed text-status-err">
              {run.error}
            </pre>
          </div>
        )}
        <p className="text-[13px] text-muted-foreground">
          The run {run.status === "cancelled" ? "was cancelled" : "failed"} —
          no results were saved.
        </p>
      </div>
    )
  }

  if (results.isLoading) return <Skeleton className="h-40 w-full" />
  if (results.isError)
    return (
      <p className="text-[13px] text-status-err">Could not load results.</p>
    )

  return (
    <div className="max-w-3xl space-y-4">
      <RunFacts run={run} />
      <MetricsTable rows={results.data?.table ?? []} />
    </div>
  )
}

function RunFacts({ run }: { run: RunDetail }) {
  const duration =
    run.started_at && run.finished_at
      ? formatDuration(
          new Date(run.finished_at).getTime() -
            new Date(run.started_at).getTime()
        )
      : null
  return (
    <div className="flex flex-wrap gap-x-8 gap-y-2 rounded-lg border bg-card px-4 py-3">
      <Fact label="Model" value={run.model} mono />
      <Fact label="Tasks" value={String(run.tasks.length)} mono />
      {duration && <Fact label="Duration" value={duration} mono />}
      <Fact
        label="Finished"
        value={
          run.finished_at ? new Date(run.finished_at).toLocaleString() : "—"
        }
      />
    </div>
  )
}

function Fact({
  label,
  value,
  mono,
}: {
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div>
      <div className="section-label">{label}</div>
      <div className={cn("mt-0.5 text-[13px]", mono && "data-mono")}>{value}</div>
    </div>
  )
}

function formatDuration(ms: number): string {
  const s = Math.round(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

function LogsTab({ runId }: { runId: string }) {
  const logs = useRunLogs(runId)
  if (logs.isLoading) return <Skeleton className="h-40 w-full" />
  if (!logs.data)
    return (
      <p className="text-[13px] text-muted-foreground">
        No log file was saved for this run.
      </p>
    )
  return (
    <pre className="max-h-[70vh] overflow-auto rounded-lg border bg-card p-3 font-mono text-[11px] leading-relaxed">
      {logs.data}
    </pre>
  )
}
