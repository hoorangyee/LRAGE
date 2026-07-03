import { ArrowDown, ArrowUp } from "lucide-react"

import type { MetricRow } from "@/api/types"
import { useSortableColumns } from "@/hooks/useSortableColumns"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"

const ACCESSORS = {
  task: (r: MetricRow) => r.task,
  n_shot: (r: MetricRow) => r.n_shot,
  metric: (r: MetricRow) => r.metric,
  value: (r: MetricRow) => r.value,
  stderr: (r: MetricRow) => r.stderr,
}

const COLUMNS: Array<{ key: keyof typeof ACCESSORS; label: string; numeric?: boolean }> = [
  { key: "task", label: "Task" },
  { key: "n_shot", label: "n-shot", numeric: true },
  { key: "metric", label: "Metric" },
  { key: "value", label: "Value", numeric: true },
  { key: "stderr", label: "± stderr", numeric: true },
]

export function MetricsTable({ rows }: { rows: MetricRow[] }) {
  const { sorted, sort, toggleSort } = useSortableColumns(rows, ACCESSORS)

  if (rows.length === 0) {
    return (
      <p className="text-[13px] text-muted-foreground">
        No aggregate metrics were produced by this run.
      </p>
    )
  }

  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            {COLUMNS.map((col) => (
              <TableHead
                key={col.key}
                className={cn(col.numeric && "text-right")}
              >
                <button
                  type="button"
                  onClick={() => toggleSort(col.key)}
                  className={cn(
                    "inline-flex items-center gap-1 hover:text-foreground",
                    col.numeric && "flex-row-reverse"
                  )}
                >
                  {col.label}
                  {sort?.key === col.key &&
                    (sort.direction === "asc" ? (
                      <ArrowUp className="size-3" />
                    ) : (
                      <ArrowDown className="size-3" />
                    ))}
                </button>
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((row, i) => (
            <TableRow key={`${row.task}-${row.metric}-${i}`}>
              <TableCell className="font-mono text-xs">{row.task}</TableCell>
              <TableCell className="data-mono text-right text-xs">
                {row.n_shot}
              </TableCell>
              <TableCell className="font-mono text-xs text-muted-foreground">
                {row.metric}
              </TableCell>
              <TableCell className="data-mono text-right text-xs font-medium text-brass">
                {row.value.toFixed(4)}
              </TableCell>
              <TableCell className="data-mono text-right text-xs text-muted-foreground">
                {row.stderr != null ? row.stderr.toFixed(4) : "—"}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
