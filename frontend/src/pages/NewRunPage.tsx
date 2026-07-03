import { useSearchParams } from "react-router-dom"

import { useRun } from "@/api/queries"
import { PageHeader } from "@/components/layout/PageHeader"
import { RunConfigForm } from "@/components/run-form/RunConfigForm"
import { Skeleton } from "@/components/ui/skeleton"
import { fromRunConfig } from "@/schemas/runConfig"

export function NewRunPage() {
  const [params] = useSearchParams()
  const fromId = params.get("from") ?? undefined
  const source = useRun(fromId)

  if (fromId && source.isLoading) {
    return (
      <div>
        <PageHeader title="New run" description={`Duplicating ${fromId}…`} />
        <div className="p-6">
          <Skeleton className="h-40 w-full max-w-3xl" />
        </div>
      </div>
    )
  }

  const initial =
    fromId && source.data
      ? fromRunConfig(source.data.config, source.data.name)
      : undefined

  return (
    <div>
      <PageHeader
        title="New run"
        description={
          initial
            ? `Duplicated from ${fromId} — API keys are not copied.`
            : "Configure and start an evaluation."
        }
      />
      <RunConfigForm key={fromId ?? "blank"} initial={initial} />
    </div>
  )
}
