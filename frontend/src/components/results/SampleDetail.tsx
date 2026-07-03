import { useState } from "react"
import { Check, Copy } from "lucide-react"

import type { SampleDetailResponse } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

/** Extract the first prompt string from the `arguments` structure.
 * EvaluationTracker saves it as {"gen_args_0": {"arg_0": prompt, ...}};
 * in-memory results use nested arrays — handle both. */
function extractPrompt(args: unknown): string {
  let node: unknown = args
  if (node && typeof node === "object" && !Array.isArray(node)) {
    const genArgs = (node as Record<string, unknown>)["gen_args_0"]
    if (genArgs && typeof genArgs === "object") {
      node = (genArgs as Record<string, unknown>)["arg_0"] ?? genArgs
    }
  }
  while (Array.isArray(node) && node.length) node = node[0]
  return typeof node === "string" ? node : JSON.stringify(node ?? "", null, 2)
}

function extractResponse(resps: unknown): string {
  let node: unknown = resps
  while (Array.isArray(node) && node.length) node = node[0]
  return typeof node === "string" ? node : JSON.stringify(node ?? "", null, 2)
}

const DOCS_START = "<DOCUMENTS START>"
const DOCS_END = "<DOCUMENTS END>"

function splitRetrievedDocs(prompt: string): {
  docs: string | null
  rest: string
} {
  const start = prompt.indexOf(DOCS_START)
  const end = prompt.indexOf(DOCS_END)
  if (start === -1 || end === -1 || end < start) return { docs: null, rest: prompt }
  return {
    docs: prompt.slice(start + DOCS_START.length, end).trim(),
    rest: (prompt.slice(0, start) + prompt.slice(end + DOCS_END.length)).trim(),
  }
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      className="size-6"
      aria-label="Copy"
      onClick={async () => {
        await navigator.clipboard.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      }}
    >
      {copied ? (
        <Check className="size-3 text-status-ok" />
      ) : (
        <Copy className="size-3" />
      )}
    </Button>
  )
}

function Block({
  title,
  text,
  collapsible = false,
  tone,
}: {
  title: string
  text: string
  collapsible?: boolean
  tone?: "ok" | "err"
}) {
  const [expanded, setExpanded] = useState(false)
  const long = collapsible && text.split("\n").length > 12
  const shown =
    long && !expanded ? text.split("\n").slice(0, 12).join("\n") + "\n…" : text

  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className="section-label">{title}</span>
        <CopyButton text={text} />
      </div>
      <pre
        className={cn(
          "overflow-x-auto whitespace-pre-wrap break-words rounded-lg border bg-card p-3 font-mono text-[11px] leading-relaxed",
          tone === "ok" && "border-status-ok/30",
          tone === "err" && "border-status-err/30"
        )}
      >
        {shown}
      </pre>
      {long && (
        <button
          type="button"
          className="mt-1 text-[11px] text-brass hover:underline"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Collapse" : "Show full prompt"}
        </button>
      )}
    </div>
  )
}

export function SampleDetail({ detail }: { detail: SampleDetailResponse }) {
  const sample = detail.sample
  const prompt = extractPrompt(sample.arguments)
  const { docs, rest } = splitRetrievedDocs(prompt)
  const response = extractResponse(sample.filtered_resps ?? sample.resps)
  const target =
    typeof sample.target === "string"
      ? sample.target
      : JSON.stringify(sample.target)
  const correct = Object.values(detail.metrics).some((v) => v === 1)
  const incorrect = Object.values(detail.metrics).some((v) => v === 0)

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="data-mono text-xs text-muted-foreground">
          doc {sample.doc_id}
        </span>
        {Object.entries(detail.metrics).map(([k, v]) => (
          <Badge
            key={k}
            variant="outline"
            className={cn(
              "data-mono font-normal",
              v === 1 && "border-status-ok/40 text-status-ok",
              v === 0 && "border-status-err/40 text-status-err"
            )}
          >
            {k} {Number.isInteger(v) ? v : v.toFixed(4)}
          </Badge>
        ))}
        {sample.Rating != null && (
          <Badge variant="outline" className="data-mono border-brass/40 font-normal text-brass">
            judge {sample.Rating}/5
          </Badge>
        )}
      </div>

      {docs && <Block title="Retrieved documents" text={docs} collapsible />}
      <Block title="Prompt" text={rest} collapsible />

      <div className="grid grid-cols-2 gap-3">
        <Block
          title="Response"
          text={response}
          tone={correct ? "ok" : incorrect ? "err" : undefined}
        />
        <Block title="Target" text={target} />
      </div>

      {sample.Explanation && (
        <div>
          <div className="mb-1 flex items-center justify-between">
            <span className="section-label">Judge explanation</span>
            <CopyButton text={sample.Explanation} />
          </div>
          <div className="rounded-lg border border-brass/30 bg-card p-3 text-[13px] leading-relaxed">
            {sample.Explanation}
          </div>
        </div>
      )}
    </div>
  )
}
