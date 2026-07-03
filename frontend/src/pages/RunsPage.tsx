import { Plus } from "lucide-react"
import { Link, useNavigate } from "react-router-dom"

import { useRuns } from "@/api/queries"
import { PageHeader } from "@/components/layout/PageHeader"
import { RunStatusBadge } from "@/components/runs/RunStatusBadge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export function RunsPage() {
  const runs = useRuns()
  const navigate = useNavigate()

  return (
    <div>
      <PageHeader
        title="Runs"
        description="Evaluation history."
        actions={
          <Button asChild size="sm" className="gap-1.5">
            <Link to="/new">
              <Plus className="size-3.5" /> New run
            </Link>
          </Button>
        }
      />
      <div className="p-6">
        {runs.isLoading && <Skeleton className="h-24 w-full" />}
        {runs.isError && (
          <p className="text-[13px] text-status-err">
            Could not load runs from the server.
          </p>
        )}
        {runs.data && runs.data.length === 0 && (
          <p className="text-[13px] text-muted-foreground">
            No runs yet. Start one from{" "}
            <Link to="/new" className="text-brass underline-offset-2 hover:underline">
              New run
            </Link>
            .
          </p>
        )}
        {runs.data && runs.data.length > 0 && (
          <div className="rounded-lg border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Run</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Tasks</TableHead>
                  <TableHead>Started</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {runs.data.map((run) => (
                  <TableRow
                    key={run.run_id}
                    className="cursor-pointer"
                    onClick={() => navigate(`/runs/${run.run_id}`)}
                  >
                    <TableCell>
                      <div className="font-medium">
                        {run.name || "Untitled run"}
                      </div>
                      <div className="font-mono text-[11px] text-muted-foreground">
                        {run.run_id}
                      </div>
                    </TableCell>
                    <TableCell>
                      <RunStatusBadge status={run.status} />
                    </TableCell>
                    <TableCell className="font-mono text-xs">
                      {run.model}
                    </TableCell>
                    <TableCell className="max-w-56">
                      <span
                        className="block truncate font-mono text-xs"
                        title={run.tasks.join(", ")}
                      >
                        {run.tasks.join(", ")}
                      </span>
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {formatWhen(run.started_at ?? run.created_at)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>
    </div>
  )
}

function formatWhen(iso: string): string {
  const date = new Date(iso)
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}
