import { useState } from "react"

import type { RunConfig } from "@/api/types"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"

function Row({ label, value }: { label: string; value: unknown }) {
  if (value === null || value === undefined || value === "") return null
  return (
    <div className="flex items-baseline justify-between gap-6 py-1">
      <span className="shrink-0 text-xs text-muted-foreground">{label}</span>
      <span className="data-mono break-all text-right text-xs">
        {typeof value === "boolean" ? (value ? "yes" : "no") : String(value)}
      </span>
    </div>
  )
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="section-label mb-1">{title}</h3>
      <div className="rounded-lg border bg-card px-3 py-1.5">{children}</div>
    </div>
  )
}

export function ConfigView({ config }: { config: RunConfig }) {
  const [showRaw, setShowRaw] = useState(false)

  return (
    <div className="max-w-2xl space-y-4">
      <div className="flex justify-end">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowRaw((v) => !v)}
          className="text-xs text-muted-foreground"
        >
          {showRaw ? "Grouped view" : "Raw JSON"}
        </Button>
      </div>

      {showRaw ? (
        <pre className="overflow-x-auto rounded-lg border bg-card p-3 font-mono text-[11px] leading-relaxed">
          {JSON.stringify(config, null, 2)}
        </pre>
      ) : (
        <div className="space-y-4">
          <Group title="Model">
            <Row label="Backend" value={config.model} />
            <Row label="Model args" value={config.model_args} />
            <Row label="Device" value={config.device ?? "auto"} />
            <Row label="Batch size" value={config.batch_size} />
          </Group>

          <Group title="Tasks">
            <Row label="Tasks" value={config.tasks.join(", ")} />
            <Row label="Few-shot" value={config.num_fewshot ?? 0} />
            <Row label="Limit" value={config.limit ?? "all"} />
          </Group>

          <Group title="Generation">
            <Row label="Max new tokens" value={config.gen_kwargs?.max_gen_toks} />
            <Row label="Temperature" value={config.gen_kwargs?.temperature} />
            <Row label="Sampling" value={config.gen_kwargs?.do_sample ?? false} />
            <Row label="System instruction" value={config.system_instruction} />
            <Row label="Chat template" value={config.apply_chat_template} />
          </Group>

          <Group title="Retrieval">
            <Row label="Enabled" value={config.retrieval.retrieve_docs} />
            {config.retrieval.retrieve_docs && (
              <>
                <Row label="Retriever" value={config.retrieval.retriever} />
                <Row label="Top-k" value={config.retrieval.top_k} />
                <Row label="Retriever args" value={config.retrieval.retriever_args} />
                <Row label="Rerank" value={config.retrieval.rerank} />
                {config.retrieval.rerank && (
                  <Row label="Reranker args" value={config.retrieval.reranker_args} />
                )}
              </>
            )}
          </Group>

          {config.judge_model && (
            <Group title="LLM-as-a-judge">
              <Row label="Judge backend" value={config.judge_model} />
              <Row label="Judge args" value={config.judge_model_args} />
            </Group>
          )}

          <Separator />
          <Group title="Seeds">
            <Row label="python / numpy / torch / fewshot" value={
              `${config.random_seed} / ${config.numpy_random_seed} / ${config.torch_random_seed} / ${config.fewshot_random_seed}`
            } />
          </Group>
        </div>
      )}
    </div>
  )
}
