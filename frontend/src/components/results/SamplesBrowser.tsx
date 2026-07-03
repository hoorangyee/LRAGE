import { useSearchParams } from "react-router-dom"
import { ChevronLeft, ChevronRight } from "lucide-react"

import { useSample, useSamples } from "@/api/queries"
import type { SampleListItem } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { SampleDetail } from "./SampleDetail"

export function SamplesBrowser({ runId }: { runId: string }) {
  const [params, setParams] = useSearchParams()
  const task = params.get("task") ?? undefined
  const page = Number(params.get("page") ?? "1")
  const q = params.get("q") ?? ""
  const incorrectOnly = params.get("incorrect") === "1"
  const docParam = params.get("doc")
  const docId = docParam != null ? Number(docParam) : null

  const setParam = (key: string, value: string | null) => {
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        if (value === null || value === "") next.delete(key)
        else next.set(key, value)
        if (key !== "doc" && key !== "page") next.delete("page")
        return next
      },
      { replace: true }
    )
  }

  const samples = useSamples(runId, { task, page, q, incorrectOnly })
  const activeTask = samples.data?.task ?? null
  const detail = useSample(runId, activeTask, docId)

  if (samples.isLoading) return <Skeleton className="h-64 w-full" />
  if (samples.isError)
    return (
      <p className="text-[13px] text-status-err">Could not load samples.</p>
    )
  const data = samples.data!
  if (data.tasks.length === 0)
    return (
      <p className="text-[13px] text-muted-foreground">
        No per-sample records were saved for this run (log_samples was off).
      </p>
    )

  const pageSize = data.page_size ?? 50
  const totalPages = Math.max(1, Math.ceil(data.total / pageSize))

  return (
    <div className="flex gap-4">
      {/* Master: filters + list */}
      <div className="w-[400px] shrink-0 space-y-3">
        <div className="flex gap-2">
          {data.tasks.length > 1 && (
            <Select
              value={data.task ?? undefined}
              onValueChange={(v) => setParam("task", v)}
            >
              <SelectTrigger className="w-44">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {data.tasks.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          <Input
            defaultValue={q}
            placeholder="Search text…"
            className="flex-1 text-xs"
            onKeyDown={(e) => {
              if (e.key === "Enter")
                setParam("q", (e.target as HTMLInputElement).value)
            }}
          />
        </div>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <Switch
            checked={incorrectOnly}
            onCheckedChange={(v) => setParam("incorrect", v ? "1" : null)}
          />
          Incorrect only
        </label>

        <div className="overflow-hidden rounded-lg border">
          {data.items.length === 0 && (
            <p className="p-3 text-xs text-muted-foreground">
              No samples match the filters.
            </p>
          )}
          {data.items.map((item) => (
            <SampleRow
              key={item.doc_id}
              item={item}
              active={item.doc_id === docId}
              onClick={() => setParam("doc", String(item.doc_id))}
            />
          ))}
        </div>

        <div className="flex items-center justify-between">
          <span className="text-[11px] text-muted-foreground">
            {data.total.toLocaleString()} samples · page {data.page}/{totalPages}
          </span>
          <div className="flex gap-1">
            <Button
              variant="outline"
              size="icon"
              className="size-7"
              disabled={data.page <= 1}
              onClick={() => setParam("page", String(data.page - 1))}
              aria-label="Previous page"
            >
              <ChevronLeft className="size-3.5" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="size-7"
              disabled={data.page >= totalPages}
              onClick={() => setParam("page", String(data.page + 1))}
              aria-label="Next page"
            >
              <ChevronRight className="size-3.5" />
            </Button>
          </div>
        </div>
      </div>

      {/* Detail */}
      <div className="min-w-0 flex-1">
        {docId == null ? (
          <p className="pt-8 text-center text-[13px] text-muted-foreground">
            Pick a sample from the list to inspect its prompt, response, and
            judge output.
          </p>
        ) : detail.isLoading ? (
          <Skeleton className="h-64 w-full" />
        ) : detail.isError || !detail.data ? (
          <p className="text-[13px] text-status-err">
            Could not load this sample.
          </p>
        ) : (
          <SampleDetail detail={detail.data} />
        )}
      </div>
    </div>
  )
}

function SampleRow({
  item,
  active,
  onClick,
}: {
  item: SampleListItem
  active: boolean
  onClick: () => void
}) {
  const wrong = Object.values(item.metrics).some((v) => v === 0)
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "block w-full border-b px-3 py-2 text-left last:border-b-0 hover:bg-accent",
        active && "bg-accent"
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="data-mono text-[11px] text-muted-foreground">
          #{item.doc_id}
        </span>
        <span className="flex gap-1">
          {Object.entries(item.metrics).map(([k, v]) => (
            <Badge
              key={k}
              variant="outline"
              className={cn(
                "data-mono px-1 py-0 text-[10px] font-normal",
                v === 1 && "border-status-ok/40 text-status-ok",
                v === 0 && "border-status-err/40 text-status-err"
              )}
            >
              {k[0]}={Number.isInteger(v) ? v : v.toFixed(2)}
            </Badge>
          ))}
          {item.rating != null && (
            <Badge
              variant="outline"
              className="data-mono border-brass/40 px-1 py-0 text-[10px] font-normal text-brass"
            >
              J{item.rating}
            </Badge>
          )}
        </span>
      </div>
      <p
        className={cn(
          "mt-1 truncate text-xs",
          wrong ? "text-foreground" : "text-muted-foreground"
        )}
      >
        {item.input || item.target}
      </p>
    </button>
  )
}
