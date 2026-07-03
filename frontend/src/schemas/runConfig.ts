import { z } from "zod"

import { isValidArgsString } from "@/lib/argsString"
import type { RunConfig, RunSubmission } from "@/api/types"

const argsField = (message: string) =>
  z.string().refine((v) => v.trim() === "" || isValidArgsString(v), {
    message: "Must be key=value pairs separated by commas",
  }).refine((v) => v.trim() !== "", { message })

export const runFormSchema = z
  .object({
    name: z.string().max(200),
    tasks: z.array(z.string()).min(1, "Pick at least one task"),
    model: z.object({
      type: z.string().min(1, "Pick a model backend"),
      preset: z.string().nullable(),
      args: argsField("Model args are required"),
    }),
    generation: z.object({
      max_gen_toks: z.coerce.number().int().min(1).max(8192),
      temperature: z.coerce.number().min(0).max(2),
      do_sample: z.boolean(),
      system_instruction: z.string(),
    }),
    retrieval: z.object({
      enabled: z.boolean(),
      top_k: z.coerce.number().int().min(1).max(100),
      preset: z.string().nullable(),
      args: z.string(),
    }),
    reranking: z.object({
      enabled: z.boolean(),
      preset: z.string().nullable(),
      args: z.string(),
    }),
    judge: z.object({
      enabled: z.boolean(),
      type: z.string(),
      preset: z.string().nullable(),
      args: z.string(),
    }),
    advanced: z.object({
      num_fewshot: z.coerce.number().int().min(0).max(64),
      limit: z.string(), // empty = no limit; number otherwise
      batch_size: z.string(), // "auto" or an integer
      device: z.string(), // "auto" = let the server decide
      apply_chat_template: z.boolean(),
      fewshot_as_multiturn: z.boolean(),
      log_samples: z.boolean(),
      random_seed: z.coerce.number().int(),
      numpy_random_seed: z.coerce.number().int(),
      torch_random_seed: z.coerce.number().int(),
      fewshot_random_seed: z.coerce.number().int(),
    }),
    api_keys: z.object({
      openai_api_key: z.string(),
      hf_token: z.string(),
    }),
  })
  .superRefine((v, ctx) => {
    if (v.retrieval.enabled) {
      if (!v.retrieval.args.trim() || !isValidArgsString(v.retrieval.args)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["retrieval", "args"],
          message: "Retriever args are required when retrieval is on",
        })
      }
    }
    if (v.reranking.enabled) {
      if (!v.retrieval.enabled) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["reranking", "enabled"],
          message: "Reranking requires retrieval",
        })
      }
      if (!v.reranking.args.trim() || !isValidArgsString(v.reranking.args)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["reranking", "args"],
          message: "Reranker args are required when reranking is on",
        })
      }
    }
    if (v.judge.enabled) {
      if (!v.judge.type) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["judge", "type"],
          message: "Pick a judge backend",
        })
      }
      if (!v.judge.args.trim() || !isValidArgsString(v.judge.args)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["judge", "args"],
          message: "Judge model args are required",
        })
      }
    }
    if (v.advanced.limit.trim() !== "" && !(Number(v.advanced.limit) > 0)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["advanced", "limit"],
        message: "Limit must be a positive number",
      })
    }
    const bs = v.advanced.batch_size.trim()
    if (bs !== "auto" && !(Number.isInteger(Number(bs)) && Number(bs) >= 1)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["advanced", "batch_size"],
        message: 'Batch size must be "auto" or a positive integer',
      })
    }
  })

/** What lives in the form fields (numeric inputs may be strings). */
export type RunFormInput = z.input<typeof runFormSchema>
/** What the resolver hands to onSubmit after validation/coercion. */
export type RunFormValues = z.output<typeof runFormSchema>

export const defaultRunFormValues: RunFormValues = {
  name: "",
  tasks: [],
  model: { type: "", preset: null, args: "" },
  generation: {
    max_gen_toks: 512,
    temperature: 0.5,
    do_sample: false,
    system_instruction: "",
  },
  retrieval: { enabled: false, top_k: 3, preset: null, args: "" },
  reranking: { enabled: false, preset: null, args: "" },
  judge: { enabled: false, type: "", preset: null, args: "" },
  advanced: {
    num_fewshot: 0,
    limit: "",
    batch_size: "auto",
    device: "auto",
    apply_chat_template: false,
    fewshot_as_multiturn: false,
    log_samples: true,
    random_seed: 0,
    numpy_random_seed: 1234,
    torch_random_seed: 1234,
    fewshot_random_seed: 1234,
  },
  api_keys: { openai_api_key: "", hf_token: "" },
}

export function toRunSubmission(v: RunFormValues): RunSubmission {
  const config: RunConfig = {
    model: v.model.type,
    model_args: v.model.args.trim(),
    tasks: v.tasks,
    judge_model: v.judge.enabled ? v.judge.type : null,
    judge_model_args: v.judge.enabled ? v.judge.args.trim() : null,
    num_fewshot: v.advanced.num_fewshot,
    fewshot_as_multiturn: v.advanced.fewshot_as_multiturn,
    batch_size:
      v.advanced.batch_size.trim() === "auto"
        ? "auto"
        : Number(v.advanced.batch_size),
    device: v.advanced.device === "auto" ? null : v.advanced.device,
    limit: v.advanced.limit.trim() === "" ? null : Number(v.advanced.limit),
    cache_requests: false,
    gen_kwargs: {
      max_gen_toks: v.generation.max_gen_toks,
      temperature: v.generation.temperature,
      do_sample: v.generation.do_sample,
    },
    system_instruction: v.generation.system_instruction.trim() || null,
    apply_chat_template: v.advanced.apply_chat_template,
    retrieval: {
      retrieve_docs: v.retrieval.enabled,
      top_k: v.retrieval.top_k,
      retriever: v.retrieval.enabled ? "pyserini" : null,
      retriever_args: v.retrieval.enabled ? v.retrieval.args.trim() : null,
      rerank: v.reranking.enabled,
      reranker: v.reranking.enabled ? "rerankers" : null,
      reranker_args: v.reranking.enabled ? v.reranking.args.trim() : null,
    },
    log_samples: v.advanced.log_samples,
    predict_only: false,
    random_seed: v.advanced.random_seed,
    numpy_random_seed: v.advanced.numpy_random_seed,
    torch_random_seed: v.advanced.torch_random_seed,
    fewshot_random_seed: v.advanced.fewshot_random_seed,
  }

  const apiKeys: Record<string, string> = {}
  if (v.api_keys.openai_api_key.trim())
    apiKeys.openai_api_key = v.api_keys.openai_api_key.trim()
  if (v.api_keys.hf_token.trim()) apiKeys.hf_token = v.api_keys.hf_token.trim()

  return {
    name: v.name.trim() || undefined,
    config,
    api_keys: Object.keys(apiKeys).length ? apiKeys : undefined,
  }
}

/** Inverse of toRunSubmission — prefill the form from a past run's config
 * ("Duplicate run"). Presets are cleared; the args strings carry the state. */
export function fromRunConfig(
  config: import("@/api/types").RunConfig,
  name?: string | null
): RunFormValues {
  return {
    name: name ?? "",
    tasks: config.tasks,
    model: { type: config.model, preset: null, args: config.model_args },
    generation: {
      max_gen_toks: config.gen_kwargs?.max_gen_toks ?? 512,
      temperature: config.gen_kwargs?.temperature ?? 0.5,
      do_sample: config.gen_kwargs?.do_sample ?? false,
      system_instruction: config.system_instruction ?? "",
    },
    retrieval: {
      enabled: config.retrieval.retrieve_docs,
      top_k: config.retrieval.top_k,
      preset: null,
      args: config.retrieval.retriever_args ?? "",
    },
    reranking: {
      enabled: config.retrieval.rerank,
      preset: null,
      args: config.retrieval.reranker_args ?? "",
    },
    judge: {
      enabled: !!config.judge_model,
      type: config.judge_model ?? "",
      preset: null,
      args: config.judge_model_args ?? "",
    },
    advanced: {
      num_fewshot: config.num_fewshot ?? 0,
      limit: config.limit != null ? String(config.limit) : "",
      batch_size: String(config.batch_size),
      device: config.device ?? "auto",
      apply_chat_template: config.apply_chat_template,
      fewshot_as_multiturn: config.fewshot_as_multiturn,
      log_samples: config.log_samples,
      random_seed: config.random_seed,
      numpy_random_seed: config.numpy_random_seed,
      torch_random_seed: config.torch_random_seed,
      fewshot_random_seed: config.fewshot_random_seed,
    },
    api_keys: { openai_api_key: "", hf_token: "" },
  }
}

/** CLI-equivalent command, shown in the summary rail. Takes raw form input
 * (numeric fields may still be strings mid-edit). */
export function toCliCommand(v: RunFormInput): string {
  const parts = [`lrage --model ${v.model.type || "?"}`]
  if (v.model.args) parts.push(`--model_args ${v.model.args}`)
  parts.push(`--tasks ${v.tasks.join(",") || "?"}`)
  const gk = `max_gen_toks=${v.generation.max_gen_toks},temperature=${v.generation.temperature},do_sample=${v.generation.do_sample ? "True" : "False"}`
  parts.push(`--gen_kwargs ${gk}`)
  if (v.retrieval.enabled) {
    parts.push("--retrieve_docs", "--retriever pyserini")
    parts.push(`--top_k ${v.retrieval.top_k}`)
    if (v.retrieval.args) parts.push(`--retriever_args ${v.retrieval.args}`)
  }
  if (v.reranking.enabled) {
    parts.push("--rerank", "--reranker rerankers")
    if (v.reranking.args) parts.push(`--reranker_args ${v.reranking.args}`)
  }
  if (v.judge.enabled && v.judge.type) {
    parts.push(`--judge_model ${v.judge.type}`)
    if (v.judge.args) parts.push(`--judge_model_args ${v.judge.args}`)
  }
  if (Number(v.advanced.num_fewshot) > 0)
    parts.push(`--num_fewshot ${v.advanced.num_fewshot}`)
  if (v.advanced.limit.trim() !== "") parts.push(`--limit ${v.advanced.limit}`)
  if (v.advanced.batch_size !== "auto")
    parts.push(`--batch_size ${v.advanced.batch_size}`)
  if (v.advanced.device !== "auto") parts.push(`--device ${v.advanced.device}`)
  if (v.advanced.apply_chat_template) parts.push("--apply_chat_template")
  return parts.join(" \\\n  ")
}
