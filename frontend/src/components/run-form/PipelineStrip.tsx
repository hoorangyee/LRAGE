import { useController, useWatch } from "react-hook-form"
import {
  Bot,
  FileSearch,
  ListOrdered,
  Scale,
  Settings2,
  SlidersHorizontal,
} from "lucide-react"

import { Switch } from "@/components/ui/switch"
import {
  NODE_ORDER,
  nodeSummary,
  type NodeKey,
} from "@/lib/experiment"
import { cn } from "@/lib/utils"
import type { RunFormInput } from "@/schemas/runConfig"

const NODE_META: Record<
  NodeKey,
  { label: string; icon: React.ComponentType<{ className?: string }> }
> = {
  tasks: { label: "Tasks", icon: ListOrdered },
  retrieve: { label: "Retrieve", icon: FileSearch },
  rerank: { label: "Rerank", icon: SlidersHorizontal },
  model: { label: "Model", icon: Bot },
  judge: { label: "Judge", icon: Scale },
  options: { label: "Options", icon: Settings2 },
}

interface PipelineStripProps {
  selected: NodeKey
  onSelect: (node: NodeKey) => void
  errorNodes: Set<NodeKey>
}

export function PipelineStrip({
  selected,
  onSelect,
  errorNodes,
}: PipelineStripProps) {
  const values = useWatch() as RunFormInput
  const retrieval = useController({ name: "retrieval.enabled" })
  const reranking = useController({ name: "reranking.enabled" })
  const judge = useController({ name: "judge.enabled" })

  const toggles: Partial<
    Record<NodeKey, { on: boolean; set: (v: boolean) => void; locked?: boolean }>
  > = {
    retrieve: { on: retrieval.field.value, set: retrieval.field.onChange },
    rerank: {
      on: reranking.field.value && values.retrieval.enabled,
      set: reranking.field.onChange,
      locked: !values.retrieval.enabled,
    },
    judge: { on: judge.field.value, set: judge.field.onChange },
  }

  const enabledFor = (node: NodeKey) => toggles[node]?.on ?? true

  return (
    <div className="flex flex-wrap items-stretch gap-y-2">
      {NODE_ORDER.map((node, i) => (
        <div key={node} className="flex items-center">
          {i > 0 && (
            <div
              aria-hidden
              className={cn(
                "h-px w-5",
                enabledFor(node) && enabledFor(NODE_ORDER[i - 1])
                  ? "bg-brass/50"
                  : "border-t border-dashed border-border"
              )}
            />
          )}
          <PipelineNode
            node={node}
            summary={nodeSummary(node, values)}
            selected={selected === node}
            enabled={enabledFor(node)}
            error={errorNodes.has(node)}
            toggle={toggles[node]}
            onSelect={() => onSelect(node)}
          />
        </div>
      ))}
      <div aria-hidden className="mx-3 my-1 w-px self-stretch bg-border" />
      <PipelineNode
        node="options"
        summary={nodeSummary("options", values)}
        selected={selected === "options"}
        enabled
        error={errorNodes.has("options")}
        onSelect={() => onSelect("options")}
      />
    </div>
  )
}

interface PipelineNodeProps {
  node: NodeKey
  summary: string
  selected: boolean
  enabled: boolean
  error: boolean
  toggle?: { on: boolean; set: (v: boolean) => void; locked?: boolean }
  onSelect: () => void
}

function PipelineNode({
  node,
  summary,
  selected,
  enabled,
  error,
  toggle,
  onSelect,
}: PipelineNodeProps) {
  const { label, icon: Icon } = NODE_META[node]

  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onSelect()
        }
      }}
      className={cn(
        "relative min-w-36 cursor-pointer rounded-lg border bg-card px-3 py-2 text-left transition-colors",
        selected
          ? "border-brass ring-1 ring-brass/40"
          : "hover:border-muted-foreground/40",
        !enabled && !selected && "opacity-60"
      )}
    >
      {error && (
        <span
          aria-label={`${label} has errors`}
          className="absolute -right-1 -top-1 size-2.5 rounded-full bg-status-err"
        />
      )}
      <div className="flex items-center justify-between gap-3">
        <span className="flex items-center gap-1.5 text-xs font-semibold">
          <Icon
            className={cn("size-3.5", enabled ? "text-brass" : "text-muted-foreground")}
          />
          {label}
        </span>
        {toggle && (
          <span
            title={toggle.locked ? "Requires retrieval" : undefined}
            onClick={(e) => e.stopPropagation()}
          >
            <Switch
              checked={toggle.on}
              onCheckedChange={toggle.set}
              disabled={toggle.locked}
              className="scale-75"
              aria-label={`Enable ${label.toLowerCase()}`}
            />
          </span>
        )}
      </div>
      <p
        className={cn(
          "mt-1 max-w-44 truncate font-mono text-[11px]",
          enabled ? "text-muted-foreground" : "text-muted-foreground/60"
        )}
        title={summary}
      >
        {summary}
      </p>
    </div>
  )
}
