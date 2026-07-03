// One editor panel per pipeline node. Panels render fields only — node
// on/off switches live in the pipeline strip.
import { useController, useFormContext, useWatch } from "react-hook-form"

import type { ModelType, PresetsResponse, TaskMeta } from "@/api/types"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import type { NodeKey } from "@/lib/experiment"
import type { RunFormInput } from "@/schemas/runConfig"
import { PresetArgsField } from "./PresetArgsField"
import { TaskMultiSelect } from "./TaskMultiSelect"

function PanelShell({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: React.ReactNode
}) {
  return (
    <section className="rounded-lg border bg-card">
      <div className="border-b px-4 py-3">
        <h2 className="text-[13px] font-semibold">{title}</h2>
        <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
      </div>
      <div className="space-y-4 px-4 py-4">{children}</div>
    </section>
  )
}

function Field({
  label,
  optional,
  error,
  children,
}: {
  label: string
  optional?: boolean
  error?: string
  children: React.ReactNode
}) {
  return (
    <div className="grid gap-1.5">
      <Label className="text-xs text-muted-foreground">
        {label} {optional && <span className="opacity-60">(optional)</span>}
      </Label>
      {children}
      {error && <p className="text-xs text-status-err">{error}</p>}
    </div>
  )
}

/* ------------------------------- Tasks -------------------------------- */

export function TasksPanel({
  tasks,
  loading,
  loadError,
}: {
  tasks: TaskMeta[]
  loading: boolean
  loadError: boolean
}) {
  const field = useController<RunFormInput, "tasks">({ name: "tasks" })
  return (
    <PanelShell
      title="Tasks"
      description="Benchmarks to evaluate on. Groups expand to all subtasks."
    >
      <TaskMultiSelect
        tasks={tasks}
        value={field.field.value}
        onChange={field.field.onChange}
        loading={loading}
      />
      {field.fieldState.error && (
        <p className="text-xs text-status-err">
          {field.fieldState.error.message}
        </p>
      )}
      {loadError && (
        <p className="text-xs text-status-err">
          Could not load the task list from the server.
        </p>
      )}
    </PanelShell>
  )
}

/* ------------------------------ Retrieve ------------------------------ */

export function RetrievePanel({
  presets,
}: {
  presets: PresetsResponse | undefined
}) {
  const { register } = useFormContext<RunFormInput>()
  const enabled = useWatch({ name: "retrieval.enabled" }) as boolean

  const retrieverPresets = (presets?.retriever_presets["pyserini"] ?? []).map(
    (args) => ({ label: retrieverPresetLabel(args), args })
  )

  if (!enabled) {
    return (
      <PanelShell
        title="Retrieve"
        description="Off — the model answers from parametric knowledge only. Turn it on from the pipeline node above."
      >
        <p className="text-xs text-muted-foreground">
          When enabled, prompts are augmented with documents fetched from a
          Pyserini index (BM25, dense, or hybrid).
        </p>
      </PanelShell>
    )
  }

  return (
    <PanelShell
      title="Retrieve"
      description="Augment prompts with documents from a Pyserini index."
    >
      <div className="grid grid-cols-2 gap-3">
        <Field label="Retriever">
          <Input value="pyserini" disabled className="font-mono text-xs" />
        </Field>
        <Field label="Top-k documents">
          <Input
            type="number"
            min={1}
            max={100}
            className="font-mono text-xs"
            {...register("retrieval.top_k")}
          />
        </Field>
      </div>
      <PresetArgsField
        presetName="retrieval.preset"
        argsName="retrieval.args"
        presets={retrieverPresets}
        argsLabel="Retriever args"
        placeholder="retriever_type=bm25,bm25_index_path=…"
      />
    </PanelShell>
  )
}

/** Compact label for a retriever args preset, e.g. "bm25 · pile-of-law-mini". */
function retrieverPresetLabel(args: string): string {
  const dict = Object.fromEntries(
    args.split(",").map((kv) => {
      const [k, ...rest] = kv.split("=")
      return [k, rest.join("=")]
    })
  )
  const type = dict["retriever_type"] ?? "?"
  const index =
    dict["faiss_index_path"] ?? dict["bm25_index_path"] ?? "unknown index"
  const indexName = index.split("/").pop() || index
  const encoder = dict["encoder_path"]
    ? ` · ${dict["encoder_path"].split("/").pop()}`
    : ""
  return `${type} · ${indexName}${encoder}`
}

/* ------------------------------- Rerank ------------------------------- */

export function RerankPanel({
  presets,
}: {
  presets: PresetsResponse | undefined
}) {
  const retrievalOn = useWatch({ name: "retrieval.enabled" }) as boolean
  const enabled = useWatch({ name: "reranking.enabled" }) as boolean

  const rerankerPresets = (presets?.reranker_presets ?? []).map((args) => ({
    label: args.replace("reranker_type=", ""),
    args,
  }))

  if (!retrievalOn || !enabled) {
    return (
      <PanelShell
        title="Rerank"
        description={
          retrievalOn
            ? "Off — retrieved documents keep their retriever order."
            : "Requires retrieval to be enabled first."
        }
      >
        <p className="text-xs text-muted-foreground">
          When enabled, a reranker (ColBERT, cross-encoder, or T5) reorders
          the retrieved documents before they enter the prompt.
        </p>
      </PanelShell>
    )
  }

  return (
    <PanelShell
      title="Rerank"
      description="Reorder retrieved documents before prompting."
    >
      <PresetArgsField
        presetName="reranking.preset"
        argsName="reranking.args"
        presets={rerankerPresets}
        argsLabel="Reranker args"
        placeholder="reranker_type=colbert"
      />
    </PanelShell>
  )
}

/* -------------------------------- Model ------------------------------- */

export function ModelPanel({
  modelTypes,
  presets,
}: {
  modelTypes: ModelType[]
  presets: PresetsResponse | undefined
}) {
  const { register, setValue } = useFormContext<RunFormInput>()
  const type = useController({ name: "model.type" })
  const doSample = useController({ name: "generation.do_sample" })
  const currentType = useWatch({ name: "model.type" }) as string
  const sampling = useWatch({ name: "generation.do_sample" }) as boolean

  const typePresets = presets?.model_presets[currentType] ?? []

  return (
    <PanelShell
      title="Model"
      description="The model under evaluation and its decoding parameters."
    >
      <Field label="Backend" error={type.fieldState.error?.message}>
        <Select
          value={type.field.value || undefined}
          onValueChange={(v) => {
            type.field.onChange(v)
            setValue("model.preset", null)
          }}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Pick a model backend" />
          </SelectTrigger>
          <SelectContent>
            {modelTypes.map((m) => (
              <SelectItem key={m.key} value={m.key}>
                {m.label}
                <span className="ml-1.5 font-mono text-[11px] text-muted-foreground">
                  {m.key}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>

      <PresetArgsField
        presetName="model.preset"
        argsName="model.args"
        presets={typePresets}
        argsLabel="Model args"
        placeholder="pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=auto"
      />

      <div className="grid grid-cols-3 gap-3 border-t pt-4">
        <Field label="Max new tokens">
          <Input
            type="number"
            min={1}
            max={8192}
            className="font-mono text-xs"
            {...register("generation.max_gen_toks")}
          />
        </Field>
        <Field label="Temperature">
          <Input
            type="number"
            step="0.1"
            min={0}
            max={2}
            disabled={!sampling}
            className="font-mono text-xs"
            {...register("generation.temperature")}
          />
        </Field>
        <Field label="Sampling">
          <div className="flex h-9 items-center">
            <Switch
              checked={doSample.field.value}
              onCheckedChange={doSample.field.onChange}
              aria-label="Enable sampling"
            />
          </div>
        </Field>
      </div>
      <Field label="System instruction" optional>
        <Textarea
          rows={3}
          placeholder="You are a legal assistant…"
          {...register("generation.system_instruction")}
        />
      </Field>

      <div className="grid grid-cols-2 gap-3 border-t pt-4">
        <Field label="OpenAI API key" optional>
          <Input
            type="password"
            autoComplete="off"
            placeholder="sk-…"
            className="font-mono text-xs"
            {...register("api_keys.openai_api_key")}
          />
        </Field>
        <Field label="Hugging Face token" optional>
          <Input
            type="password"
            autoComplete="off"
            placeholder="hf_…"
            className="font-mono text-xs"
            {...register("api_keys.hf_token")}
          />
        </Field>
        <p className="col-span-2 text-[11px] text-muted-foreground">
          Keys are used for this run only and never stored on disk. The server
          also inherits keys from its own environment.
        </p>
      </div>
    </PanelShell>
  )
}

/* -------------------------------- Judge ------------------------------- */

export function JudgePanel({
  modelTypes,
  presets,
}: {
  modelTypes: ModelType[]
  presets: PresetsResponse | undefined
}) {
  const { setValue } = useFormContext<RunFormInput>()
  const enabled = useWatch({ name: "judge.enabled" }) as boolean
  const type = useController({ name: "judge.type" })
  const currentType = useWatch({ name: "judge.type" }) as string

  const typePresets = presets?.model_presets[currentType] ?? []

  if (!enabled) {
    return (
      <PanelShell
        title="Judge"
        description="Off — only tasks with conventional metrics will score."
      >
        <p className="text-xs text-muted-foreground">
          When enabled, an LLM judge scores generations on rubric tasks
          (metric: LLM-Eval) with a rating and an explanation per sample.
        </p>
      </PanelShell>
    )
  }

  return (
    <PanelShell
      title="Judge"
      description="Scores generations on rubric tasks (metric: LLM-Eval)."
    >
      <Field label="Judge backend" error={type.fieldState.error?.message}>
        <Select
          value={type.field.value || undefined}
          onValueChange={(v) => {
            type.field.onChange(v)
            setValue("judge.preset", null)
          }}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Pick a judge backend" />
          </SelectTrigger>
          <SelectContent>
            {modelTypes.map((m) => (
              <SelectItem key={m.key} value={m.key}>
                {m.label}
                <span className="ml-1.5 font-mono text-[11px] text-muted-foreground">
                  {m.key}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
      <PresetArgsField
        presetName="judge.preset"
        argsName="judge.args"
        presets={typePresets}
        argsLabel="Judge model args"
        placeholder="model=gpt-4o-mini"
      />
    </PanelShell>
  )
}

/* ------------------------------- Options ------------------------------ */

export function OptionsPanel({ devices }: { devices: string[] }) {
  const { register, getFieldState, formState } = useFormContext<RunFormInput>()
  const device = useController({ name: "advanced.device" })
  const chatTemplate = useController({ name: "advanced.apply_chat_template" })
  const multiturn = useController({ name: "advanced.fewshot_as_multiturn" })
  const logSamples = useController({ name: "advanced.log_samples" })

  const limitError = getFieldState("advanced.limit", formState).error?.message
  const batchError = getFieldState("advanced.batch_size", formState).error
    ?.message

  return (
    <PanelShell
      title="Run options"
      description="Sampling of the dataset, hardware, and reproducibility."
    >
      <div className="grid grid-cols-3 gap-3">
        <Field label="Few-shot examples">
          <Input
            type="number"
            min={0}
            max={64}
            className="font-mono text-xs"
            {...register("advanced.num_fewshot")}
          />
        </Field>
        <Field label="Limit (docs per task)" error={limitError}>
          <Input
            placeholder="all"
            className="font-mono text-xs"
            {...register("advanced.limit")}
          />
        </Field>
        <Field label="Batch size" error={batchError}>
          <Input className="font-mono text-xs" {...register("advanced.batch_size")} />
        </Field>
      </div>

      <Field label="Device">
        <Select value={device.field.value} onValueChange={device.field.onChange}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="auto">auto (server decides)</SelectItem>
            {devices.map((d) => (
              <SelectItem key={d} value={d}>
                {d}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>

      <div className="grid grid-cols-3 gap-3">
        <Field label="Apply chat template">
          <div className="flex h-9 items-center">
            <Switch
              checked={chatTemplate.field.value}
              onCheckedChange={chatTemplate.field.onChange}
            />
          </div>
        </Field>
        <Field label="Few-shot as multi-turn">
          <div className="flex h-9 items-center">
            <Switch
              checked={multiturn.field.value}
              onCheckedChange={multiturn.field.onChange}
            />
          </div>
        </Field>
        <Field label="Log per-sample results">
          <div className="flex h-9 items-center">
            <Switch
              checked={logSamples.field.value}
              onCheckedChange={logSamples.field.onChange}
            />
          </div>
        </Field>
      </div>

      <div className="grid grid-cols-4 gap-3 border-t pt-4">
        <Field label="Seed (python)">
          <Input
            type="number"
            className="font-mono text-xs"
            {...register("advanced.random_seed")}
          />
        </Field>
        <Field label="Seed (numpy)">
          <Input
            type="number"
            className="font-mono text-xs"
            {...register("advanced.numpy_random_seed")}
          />
        </Field>
        <Field label="Seed (torch)">
          <Input
            type="number"
            className="font-mono text-xs"
            {...register("advanced.torch_random_seed")}
          />
        </Field>
        <Field label="Seed (few-shot)">
          <Input
            type="number"
            className="font-mono text-xs"
            {...register("advanced.fewshot_random_seed")}
          />
        </Field>
      </div>
    </PanelShell>
  )
}

export type { NodeKey }
