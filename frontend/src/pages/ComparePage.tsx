import { useSearchParams } from "react-router-dom"
import { Check, ChevronsUpDown, X } from "lucide-react"

import { useCompare, useRuns } from "@/api/queries"
import type { CompareResponse, CompareRow, RunConfig } from "@/api/types"
import { PageHeader } from "@/components/layout/PageHeader"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"

const MAX_RUNS = 4

export function ComparePage() {
  const [params, setParams] = useSearchParams()
  const selected = (params.get("runs") ?? "").split(",").filter(Boolean)

  const setSelected = (ids: string[]) => {
    setParams(ids.length ? { runs: ids.join(",") } : {}, { replace: true })
  }

  const compare = useCompare(selected)

  return (
    <div>
      <PageHeader title="Compare" description="Metric deltas across runs." />
      <div className="space-y-5 p-6">
        <RunPicker selected={selected} onChange={setSelected} />

        {selected.length < 2 && (
          <p className="text-[13px] text-muted-foreground">
            Pick two to four completed runs. The first run is the baseline.
          </p>
        )}
        {compare.isLoading && selected.length >= 2 && (
          <Skeleton className="h-48 w-full" />
        )}
        {compare.isError && (
          <p className="text-[13px] text-status-err">
            {String(compare.error.message)}
          </p>
        )}
        {compare.data && <CompareResults data={compare.data} />}
      </div>
    </div>
  )
}

function RunPicker({
  selected,
  onChange,
}: {
  selected: string[]
  onChange: (ids: string[]) => void
}) {
  const runs = useRuns()
  const completed = (runs.data ?? []).filter((r) => r.status === "completed")

  const toggle = (id: string) => {
    if (selected.includes(id)) onChange(selected.filter((s) => s !== id))
    else if (selected.length < MAX_RUNS) onChange([...selected, id])
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="outline" className="justify-between gap-2 font-normal">
            <span className="text-muted-foreground">
              {selected.length
                ? `${selected.length} run${selected.length > 1 ? "s" : ""} selected`
                : "Pick runs to compare…"}
            </span>
            <ChevronsUpDown className="size-3.5 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[420px] p-0" align="start">
          <Command>
            <CommandInput placeholder="Filter completed runs…" />
            <CommandList>
              <CommandEmpty>No completed runs.</CommandEmpty>
              <CommandGroup>
                {completed.map((run) => (
                  <CommandItem
                    key={run.run_id}
                    value={`${run.name ?? ""} ${run.run_id} ${run.model}`}
                    onSelect={() => toggle(run.run_id)}
                  >
                    <Check
                      className={cn(
                        "size-3.5",
                        selected.includes(run.run_id)
                          ? "opacity-100"
                          : "opacity-0"
                      )}
                    />
                    <div className="min-w-0">
                      <div className="truncate text-xs">
                        {run.name || "Untitled run"}
                      </div>
                      <div className="font-mono text-[10px] text-muted-foreground">
                        {run.run_id} · {run.model}
                      </div>
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      {selected.map((id, i) => (
        <Badge key={id} variant="secondary" className="gap-1 pr-1">
          {i === 0 && (
            <span className="text-[10px] uppercase text-brass">base</span>
          )}
          <span className="font-mono text-[11px]">{id}</span>
          <button
            type="button"
            aria-label={`Remove ${id}`}
            onClick={() => onChange(selected.filter((s) => s !== id))}
            className="rounded-sm p-0.5 hover:bg-muted-foreground/20"
          >
            <X className="size-3" />
          </button>
        </Badge>
      ))}
    </div>
  )
}

function CompareResults({ data }: { data: CompareResponse }) {
  const runLabel = (runId: string) => {
    const run = data.runs.find((r) => r.run_id === runId)
    return run?.name || runId
  }
  const baseline = data.runs[0]?.run_id

  return (
    <div className="space-y-6">
      <div className="rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Task</TableHead>
              <TableHead>Metric</TableHead>
              {data.runs.map((run) => (
                <TableHead key={run.run_id} className="min-w-44 text-right">
                  <span title={run.run_id}>{runLabel(run.run_id)}</span>
                  {run.run_id === baseline && (
                    <span className="ml-1 text-[10px] uppercase text-brass">
                      base
                    </span>
                  )}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.table.map((row) => (
              <CompareTableRow
                key={`${row.task}.${row.metric}`}
                row={row}
                runIds={data.runs.map((r) => r.run_id)}
                baseline={baseline}
              />
            ))}
          </TableBody>
        </Table>
      </div>

      <ConfigDiff data={data} />
    </div>
  )
}

function CompareTableRow({
  row,
  runIds,
  baseline,
}: {
  row: CompareRow
  runIds: string[]
  baseline: string
}) {
  const values = runIds.map((id) => row.values[id]?.value ?? null)
  const max = Math.max(...values.filter((v): v is number => v != null))
  const base = row.values[baseline]?.value ?? null

  return (
    <TableRow>
      <TableCell className="font-mono text-xs">{row.task}</TableCell>
      <TableCell className="font-mono text-xs text-muted-foreground">
        {row.metric}
      </TableCell>
      {runIds.map((id, i) => {
        const cell = row.values[id]
        if (!cell)
          return (
            <TableCell key={id} className="text-right text-xs text-muted-foreground">
              —
            </TableCell>
          )
        const delta = base != null && id !== baseline ? cell.value - base : null
        const best = cell.value === max
        return (
          <TableCell key={id} className="text-right">
            <div className="flex items-center justify-end gap-2">
              {delta != null && Math.abs(delta) > 1e-9 && (
                <span
                  className={cn(
                    "data-mono text-[10px]",
                    delta > 0 ? "text-status-ok" : "text-status-err"
                  )}
                >
                  {delta > 0 ? "▲" : "▼"}
                  {Math.abs(delta).toFixed(4)}
                </span>
              )}
              <span
                className={cn(
                  "data-mono text-xs",
                  best ? "font-semibold text-brass" : "text-foreground"
                )}
              >
                {cell.value.toFixed(4)}
              </span>
            </div>
            <MetricBar value={cell.value} max={max} highlight={best} rank={i} />
          </TableCell>
        )
      })}
    </TableRow>
  )
}

function MetricBar({
  value,
  max,
  highlight,
}: {
  value: number
  max: number
  highlight: boolean
  rank: number
}) {
  const pct = max > 0 ? Math.max(2, (value / max) * 100) : 0
  return (
    <div className="mt-1 h-1 w-full rounded-full bg-muted">
      <div
        className={cn(
          "h-1 rounded-full",
          highlight ? "bg-brass" : "bg-muted-foreground/40"
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

const DIFF_FIELDS: Array<{ label: string; get: (c: RunConfig) => unknown }> = [
  { label: "Model backend", get: (c) => c.model },
  { label: "Model args", get: (c) => c.model_args },
  { label: "Tasks", get: (c) => c.tasks.join(", ") },
  { label: "Few-shot", get: (c) => c.num_fewshot ?? 0 },
  { label: "Limit", get: (c) => c.limit ?? "all" },
  { label: "Retrieval", get: (c) => c.retrieval.retrieve_docs },
  { label: "Retriever args", get: (c) => c.retrieval.retriever_args ?? "—" },
  { label: "Top-k", get: (c) => c.retrieval.top_k },
  { label: "Rerank", get: (c) => c.retrieval.rerank },
  { label: "Reranker args", get: (c) => c.retrieval.reranker_args ?? "—" },
  { label: "Judge", get: (c) => c.judge_model ?? "—" },
  { label: "Temperature", get: (c) => c.gen_kwargs?.temperature ?? "—" },
  { label: "Max new tokens", get: (c) => c.gen_kwargs?.max_gen_toks ?? "—" },
  { label: "System instruction", get: (c) => c.system_instruction ?? "—" },
]

function ConfigDiff({ data }: { data: CompareResponse }) {
  const rows = DIFF_FIELDS.map((field) => ({
    label: field.label,
    values: data.runs.map((r) => String(field.get(r.config))),
  })).filter((row) => new Set(row.values).size > 1)

  if (rows.length === 0)
    return (
      <p className="text-[13px] text-muted-foreground">
        The selected runs have identical configurations.
      </p>
    )

  return (
    <div>
      <h2 className="section-label mb-2">What differs between these runs</h2>
      <div className="rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Setting</TableHead>
              {data.runs.map((run) => (
                <TableHead key={run.run_id}>
                  {run.name || run.run_id}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.label}>
                <TableCell className="text-xs text-muted-foreground">
                  {row.label}
                </TableCell>
                {row.values.map((value, i) => (
                  <TableCell
                    key={i}
                    className="max-w-64 break-all font-mono text-xs"
                  >
                    {value}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
