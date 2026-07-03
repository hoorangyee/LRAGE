import { useMemo, useState } from "react"
import { zodResolver } from "@hookform/resolvers/zod"
import { FormProvider, useForm } from "react-hook-form"
import { useNavigate } from "react-router-dom"

import { useSubmitRun } from "@/api/mutations"
import {
  useDevices,
  usePresets,
  useRegistries,
  useTasks,
} from "@/api/queries"
import type { RunDetail } from "@/api/types"
import { useLocalDraft } from "@/hooks/useLocalDraft"
import {
  errorPaths,
  nodeForErrorPath,
  type NodeKey,
} from "@/lib/experiment"
import {
  defaultRunFormValues,
  fromRunConfig,
  runFormSchema,
  toRunSubmission,
  type RunFormInput,
  type RunFormValues,
} from "@/schemas/runConfig"
import { LaunchBar } from "./LaunchBar"
import {
  JudgePanel,
  ModelPanel,
  OptionsPanel,
  RerankPanel,
  RetrievePanel,
  TasksPanel,
} from "./panels"
import { PipelineStrip } from "./PipelineStrip"
import { StartFromRow } from "./StartFromRow"

const DRAFT_KEY = "lrage:new-run-draft"

interface RunConfigFormProps {
  /** Prefill (e.g. duplicating via /new?from=). Disables draft + start-from row. */
  initial?: RunFormValues
}

export function RunConfigForm({ initial }: RunConfigFormProps) {
  const navigate = useNavigate()
  const tasks = useTasks()
  const registries = useRegistries()
  const presets = usePresets()
  const devices = useDevices()
  const submitRun = useSubmitRun()

  const [selectedNode, setSelectedNode] = useState<NodeKey>("tasks")
  const [startedFrom, setStartedFrom] = useState<string | null>(null)

  const form = useForm<RunFormInput, unknown, RunFormValues>({
    resolver: zodResolver(runFormSchema),
    defaultValues: initial ?? defaultRunFormValues,
    mode: "onTouched",
  })
  const draft = useLocalDraft(DRAFT_KEY, form, initial === undefined)

  const presetTypes = presets.data?.model_types ?? []
  const registryTypes = registries.data?.model_types ?? []
  const modelTypes = [
    ...presetTypes,
    ...registryTypes.filter((m) => !presetTypes.some((p) => p.key === m.key)),
  ]

  const errorNodes = useMemo(() => {
    return new Set(
      errorPaths(form.formState.errors).map((path) => nodeForErrorPath(path))
    )
    // formState.errors identity changes on validation runs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.formState.errors, form.formState.submitCount, form.formState.isValid])

  const errorCount = errorPaths(form.formState.errors).length

  const onSubmit = form.handleSubmit(
    async (values) => {
      const res = await submitRun.mutateAsync(toRunSubmission(values))
      draft.clear()
      navigate(`/runs/${res.run_id}`)
    },
    (errors) => {
      const first = errorPaths(errors)[0]
      if (first) setSelectedNode(nodeForErrorPath(first))
    }
  )

  const loadFromRun = (run: RunDetail) => {
    form.reset(fromRunConfig(run.config, run.name))
    setStartedFrom(run.run_id)
    setSelectedNode("tasks")
  }

  const loadBlank = () => {
    form.reset(defaultRunFormValues)
    setStartedFrom(null)
    setSelectedNode("tasks")
  }

  return (
    <FormProvider {...form}>
      <form onSubmit={onSubmit} className="flex min-h-[calc(100vh-73px)] flex-col">
        <div className="mx-auto w-full max-w-5xl flex-1 space-y-5 px-6 py-6">
          {initial === undefined && (
            <StartFromRow
              onPick={loadFromRun}
              onBlank={loadBlank}
              startedFrom={startedFrom}
            />
          )}

          <div className="sticky top-0 z-10 -mx-6 border-y bg-background/95 px-6 py-3 backdrop-blur">
            <PipelineStrip
              selected={selectedNode}
              onSelect={setSelectedNode}
              errorNodes={errorNodes}
            />
          </div>

          {selectedNode === "tasks" && (
            <TasksPanel
              tasks={tasks.data ?? []}
              loading={tasks.isLoading}
              loadError={tasks.isError}
            />
          )}
          {selectedNode === "retrieve" && <RetrievePanel presets={presets.data} />}
          {selectedNode === "rerank" && <RerankPanel presets={presets.data} />}
          {selectedNode === "model" && (
            <ModelPanel modelTypes={modelTypes} presets={presets.data} />
          )}
          {selectedNode === "judge" && (
            <JudgePanel modelTypes={modelTypes} presets={presets.data} />
          )}
          {selectedNode === "options" && (
            <OptionsPanel devices={devices.data ?? []} />
          )}
        </div>

        <LaunchBar
          submitting={submitRun.isPending}
          submitError={
            submitRun.isError ? String(submitRun.error.message) : null
          }
          errorCount={errorCount}
        />
      </form>
    </FormProvider>
  )
}
