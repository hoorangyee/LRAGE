import { describe, expect, it } from "vitest"

import { defaultRunFormValues, type RunFormValues } from "@/schemas/runConfig"
import {
  errorPaths,
  experimentSentence,
  nodeForErrorPath,
  nodeSummary,
} from "./experiment"

function base(): RunFormValues {
  return JSON.parse(JSON.stringify(defaultRunFormValues))
}

function ragConfig(): RunFormValues {
  const v = base()
  v.tasks = ["barexam_qa", "housing_qa", "abercrombie"]
  v.model = {
    type: "huggingface",
    preset: null,
    args: "pretrained=Qwen/Qwen2.5-7B-Instruct,dtype=auto",
  }
  v.retrieval = {
    enabled: true,
    top_k: 3,
    preset: null,
    args: "retriever_type=bm25,bm25_index_path=../../bm25_indexes/pile-of-law-mini",
  }
  v.reranking = { enabled: true, preset: null, args: "reranker_type=colbert" }
  v.judge = {
    enabled: true,
    type: "openai-chat-completions",
    preset: null,
    args: "model=gpt-4o-mini",
  }
  return v
}

describe("nodeSummary", () => {
  it("summarizes each node of a RAG config", () => {
    const v = ragConfig()
    expect(nodeSummary("tasks", v)).toBe("barexam_qa +2")
    expect(nodeSummary("retrieve", v)).toBe("bm25 · pile-of-law-mini · k3")
    expect(nodeSummary("rerank", v)).toBe("colbert")
    expect(nodeSummary("model", v)).toBe("Qwen2.5-7B-Instruct")
    expect(nodeSummary("judge", v)).toBe("gpt-4o-mini")
    expect(nodeSummary("options", v)).toBe("defaults")
  })

  it("reflects off/locked states", () => {
    const v = base()
    expect(nodeSummary("tasks", v)).toBe("none selected")
    expect(nodeSummary("retrieve", v)).toBe("off")
    expect(nodeSummary("rerank", v)).toBe("needs retrieval")
    expect(nodeSummary("judge", v)).toBe("off")
  })

  it("summarizes non-default options", () => {
    const v = base()
    v.advanced.num_fewshot = 3
    v.advanced.limit = "100"
    v.advanced.device = "cuda:0"
    expect(nodeSummary("options", v)).toBe("3-shot · limit 100 · cuda:0")
  })
})

describe("experimentSentence", () => {
  it("reads as a full sentence for a RAG + judge config", () => {
    expect(experimentSentence(ragConfig())).toBe(
      "Evaluate Qwen2.5-7B-Instruct on 3 tasks with bm25 top-3 + colbert, judged by gpt-4o-mini"
    )
  })

  it("stays minimal for a plain config", () => {
    const v = base()
    v.tasks = ["abercrombie"]
    v.model = { type: "openai-chat-completions", preset: "gpt-4o-mini", args: "model=gpt-4o-mini" }
    expect(experimentSentence(v)).toBe("Evaluate gpt-4o-mini on abercrombie")
  })
})

describe("error mapping", () => {
  it("maps paths to owning nodes", () => {
    expect(nodeForErrorPath("tasks")).toBe("tasks")
    expect(nodeForErrorPath("retrieval.args")).toBe("retrieve")
    expect(nodeForErrorPath("reranking.enabled")).toBe("rerank")
    expect(nodeForErrorPath("model.args")).toBe("model")
    expect(nodeForErrorPath("generation.max_gen_toks")).toBe("model")
    expect(nodeForErrorPath("judge.type")).toBe("judge")
    expect(nodeForErrorPath("advanced.limit")).toBe("options")
  })

  it("flattens nested RHF error objects", () => {
    const errors = {
      tasks: { message: "Pick at least one task", type: "too_small" },
      model: {
        type: { message: "Pick a model backend", type: "custom" },
        args: { message: "Model args are required", type: "custom" },
      },
    }
    expect(errorPaths(errors).sort()).toEqual([
      "model.args",
      "model.type",
      "tasks",
    ])
  })
})
