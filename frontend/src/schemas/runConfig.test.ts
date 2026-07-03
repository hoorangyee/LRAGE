import { describe, expect, it } from "vitest"

import {
  defaultRunFormValues,
  runFormSchema,
  toRunSubmission,
  type RunFormValues,
} from "./runConfig"

function base(): RunFormValues {
  return structuredClone
    ? structuredClone(defaultRunFormValues)
    : JSON.parse(JSON.stringify(defaultRunFormValues))
}

function valid(): RunFormValues {
  const v = base()
  v.tasks = ["abercrombie"]
  v.model = { type: "openai-chat-completions", preset: "gpt-4o-mini", args: "model=gpt-4o-mini" }
  return v
}

describe("runFormSchema", () => {
  it("accepts a minimal valid config", () => {
    expect(runFormSchema.safeParse(valid()).success).toBe(true)
  })

  it("requires at least one task", () => {
    const v = valid()
    v.tasks = []
    const res = runFormSchema.safeParse(v)
    expect(res.success).toBe(false)
  })

  it("requires retriever args when retrieval is on", () => {
    const v = valid()
    v.retrieval.enabled = true
    v.retrieval.args = ""
    const res = runFormSchema.safeParse(v)
    expect(res.success).toBe(false)
    expect(JSON.stringify(res.error?.issues)).toContain("retrieval")
  })

  it("requires retrieval for reranking", () => {
    const v = valid()
    v.reranking.enabled = true
    v.reranking.args = "reranker_type=colbert"
    const res = runFormSchema.safeParse(v)
    expect(res.success).toBe(false)
    expect(
      res.error?.issues.some((i) => i.path.join(".") === "reranking.enabled")
    ).toBe(true)
  })

  it("requires judge type and args when judge is on", () => {
    const v = valid()
    v.judge.enabled = true
    const res = runFormSchema.safeParse(v)
    expect(res.success).toBe(false)
  })

  it("coerces numeric strings", () => {
    const v = valid()
    // Form inputs deliver strings.
    ;(v.generation as Record<string, unknown>).max_gen_toks = "256"
    const res = runFormSchema.safeParse(v)
    expect(res.success).toBe(true)
    if (res.success) expect(res.data.generation.max_gen_toks).toBe(256)
  })

  it("validates batch_size as auto or integer", () => {
    const v = valid()
    v.advanced.batch_size = "16"
    expect(runFormSchema.safeParse(v).success).toBe(true)
    v.advanced.batch_size = "auto"
    expect(runFormSchema.safeParse(v).success).toBe(true)
    v.advanced.batch_size = "sixteen"
    expect(runFormSchema.safeParse(v).success).toBe(false)
  })
})

describe("toRunSubmission", () => {
  it("maps a plain run", () => {
    const sub = toRunSubmission(valid())
    expect(sub.config.model).toBe("openai-chat-completions")
    expect(sub.config.model_args).toBe("model=gpt-4o-mini")
    expect(sub.config.tasks).toEqual(["abercrombie"])
    expect(sub.config.retrieval.retrieve_docs).toBe(false)
    expect(sub.config.retrieval.retriever).toBeNull()
    expect(sub.config.judge_model).toBeNull()
    expect(sub.config.limit).toBeNull()
    expect(sub.config.batch_size).toBe("auto")
    expect(sub.config.device).toBeNull()
    expect(sub.api_keys).toBeUndefined()
  })

  it("maps retrieval + reranking + judge", () => {
    const v = valid()
    v.retrieval = {
      enabled: true,
      top_k: 5,
      preset: null,
      args: "retriever_type=bm25,bm25_index_path=x",
    }
    v.reranking = { enabled: true, preset: null, args: "reranker_type=colbert" }
    v.judge = {
      enabled: true,
      type: "openai-chat-completions",
      preset: null,
      args: "model=gpt-4o-mini",
    }
    v.advanced.limit = "5"
    v.advanced.batch_size = "8"
    v.api_keys.openai_api_key = "sk-test"

    const sub = toRunSubmission(v)
    expect(sub.config.retrieval).toEqual({
      retrieve_docs: true,
      top_k: 5,
      retriever: "pyserini",
      retriever_args: "retriever_type=bm25,bm25_index_path=x",
      rerank: true,
      reranker: "rerankers",
      reranker_args: "reranker_type=colbert",
    })
    expect(sub.config.judge_model).toBe("openai-chat-completions")
    expect(sub.config.limit).toBe(5)
    expect(sub.config.batch_size).toBe(8)
    expect(sub.api_keys).toEqual({ openai_api_key: "sk-test" })
  })
})
