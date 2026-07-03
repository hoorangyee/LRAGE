import { Check, Loader2 } from "lucide-react"

import type { RunConfig } from "@/api/types"
import { cn } from "@/lib/utils"

interface PhaseStep {
  key: string
  label: string
  /** phases reported by the backend that map onto this step */
  matches: string[]
}

function stepsFor(config: RunConfig): PhaseStep[] {
  const steps: PhaseStep[] = [
    {
      key: "load",
      label: "Load model",
      matches: ["loading_model", "loading_retriever", "loading_reranker"],
    },
  ]
  if (config.retrieval.retrieve_docs)
    steps.push({ key: "retrieve", label: "Retrieve", matches: ["retrieving"] })
  if (config.retrieval.rerank)
    steps.push({ key: "rerank", label: "Rerank", matches: ["reranking"] })
  steps.push({
    key: "build",
    label: "Build prompts",
    matches: ["building_requests"],
  })
  steps.push({
    key: "generate",
    label: "Generate",
    matches: ["running_requests"],
  })
  if (config.judge_model)
    steps.push({ key: "judge", label: "Judge", matches: ["judging"] })
  steps.push({ key: "save", label: "Save", matches: ["saving"] })
  return steps
}

interface PhaseStepperProps {
  config: RunConfig
  phase: string | null | undefined
}

export function PhaseStepper({ config, phase }: PhaseStepperProps) {
  const steps = stepsFor(config)
  const activeIdx = phase
    ? steps.findIndex((s) => s.matches.includes(phase))
    : -1

  return (
    <ol className="flex flex-wrap items-center gap-1">
      {steps.map((step, i) => {
        const state =
          activeIdx === -1 ? "pending" : i < activeIdx ? "done" : i === activeIdx ? "active" : "pending"
        return (
          <li key={step.key} className="flex items-center gap-1">
            {i > 0 && <span className="mx-1 h-px w-4 bg-border" aria-hidden />}
            <span
              className={cn(
                "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs",
                state === "done" && "border-status-ok/40 text-status-ok",
                state === "active" && "border-brass/50 bg-brass/10 font-medium text-brass",
                state === "pending" && "border-border text-muted-foreground"
              )}
            >
              {state === "done" && <Check className="size-3" aria-hidden />}
              {state === "active" && (
                <Loader2 className="size-3 animate-spin" aria-hidden />
              )}
              {step.label}
            </span>
          </li>
        )
      })}
    </ol>
  )
}
