// Derivations from the run form state used by the pipeline canvas:
// per-node one-line summaries and the launch-bar experiment sentence.
import { parseArgsString } from "@/lib/argsString"
import type { RunFormInput } from "@/schemas/runConfig"

export type NodeKey =
  | "tasks"
  | "retrieve"
  | "rerank"
  | "model"
  | "judge"
  | "options"

export const NODE_ORDER: NodeKey[] = [
  "tasks",
  "retrieve",
  "rerank",
  "model",
  "judge",
]

function argValue(args: string, key: string): string | null {
  const pairs = parseArgsString(args)
  return pairs?.find((p) => p.key === key)?.value ?? null
}

function basename(path: string): string {
  return path.split("/").pop() || path
}

/** Short model name from an args string: pretrained=/model= basename. */
export function modelShortName(v: RunFormInput): string | null {
  if (v.model.preset) return v.model.preset
  const args = v.model.args
  const name = argValue(args, "pretrained") ?? argValue(args, "model")
  if (name) return basename(name)
  return v.model.type ? v.model.type : null
}

export function nodeSummary(node: NodeKey, v: RunFormInput): string {
  switch (node) {
    case "tasks": {
      if (v.tasks.length === 0) return "none selected"
      if (v.tasks.length === 1) return v.tasks[0]
      return `${v.tasks[0]} +${v.tasks.length - 1}`
    }
    case "retrieve": {
      if (!v.retrieval.enabled) return "off"
      const type = argValue(v.retrieval.args, "retriever_type") ?? "pyserini"
      const index =
        argValue(v.retrieval.args, "faiss_index_path") ??
        argValue(v.retrieval.args, "bm25_index_path")
      const indexName = index ? ` · ${basename(index)}` : ""
      return `${type}${indexName} · k${v.retrieval.top_k}`
    }
    case "rerank": {
      if (!v.retrieval.enabled) return "needs retrieval"
      if (!v.reranking.enabled) return "off"
      return argValue(v.reranking.args, "reranker_type") ?? "on"
    }
    case "model": {
      const name = modelShortName(v)
      if (!name) return "not configured"
      const sampling = v.generation.do_sample
        ? ` · T${v.generation.temperature}`
        : ""
      return `${name}${sampling}`
    }
    case "judge": {
      if (!v.judge.enabled) return "off"
      if (!v.judge.type) return "pick a backend"
      const name = argValue(v.judge.args, "model") ?? v.judge.type
      return basename(name)
    }
    case "options": {
      const a = v.advanced
      const parts: string[] = []
      if (Number(a.num_fewshot) > 0) parts.push(`${a.num_fewshot}-shot`)
      if (a.limit.trim() !== "") parts.push(`limit ${a.limit}`)
      if (a.batch_size !== "auto") parts.push(`bs ${a.batch_size}`)
      if (a.device !== "auto") parts.push(a.device)
      if (a.apply_chat_template) parts.push("chat template")
      return parts.length ? parts.join(" · ") : "defaults"
    }
  }
}

/** "Evaluate qwen2.5-7b on 3 tasks with bm25 top-3 + colbert, judged by gpt-4o-mini" */
export function experimentSentence(v: RunFormInput): string {
  const model = modelShortName(v) ?? "…"
  const tasks =
    v.tasks.length === 0
      ? "…"
      : v.tasks.length === 1
        ? v.tasks[0]
        : `${v.tasks.length} tasks`
  let sentence = `Evaluate ${model} on ${tasks}`
  if (v.retrieval.enabled) {
    const type = argValue(v.retrieval.args, "retriever_type") ?? "retrieval"
    sentence += ` with ${type} top-${v.retrieval.top_k}`
    if (v.reranking.enabled) {
      const reranker = argValue(v.reranking.args, "reranker_type")
      sentence += ` + ${reranker ?? "rerank"}`
    }
  }
  if (v.judge.enabled && v.judge.type) {
    const judge = argValue(v.judge.args, "model") ?? v.judge.type
    sentence += `, judged by ${basename(judge)}`
  }
  return sentence
}

/** Map a zod/RHF error path to the pipeline node that owns the field. */
export function nodeForErrorPath(path: string): NodeKey {
  if (path.startsWith("tasks")) return "tasks"
  if (path.startsWith("retrieval")) return "retrieve"
  if (path.startsWith("reranking")) return "rerank"
  if (path.startsWith("judge")) return "judge"
  if (path.startsWith("advanced")) return "options"
  return "model" // model.*, generation.*, api_keys.*, name
}

/** Flatten RHF's nested error object into dotted paths. */
export function errorPaths(errors: object, prefix = ""): string[] {
  const out: string[] = []
  for (const [key, value] of Object.entries(errors)) {
    if (!value || typeof value !== "object") continue
    const path = prefix ? `${prefix}.${key}` : key
    if (typeof (value as { message?: unknown }).message === "string") {
      out.push(path)
    } else {
      out.push(...errorPaths(value, path))
    }
  }
  return out
}
