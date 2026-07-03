import { useState } from "react"
import { Check, Copy, Play, Terminal } from "lucide-react"
import { useController, useWatch } from "react-hook-form"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { experimentSentence } from "@/lib/experiment"
import {
  toCliCommand,
  type RunFormInput,
} from "@/schemas/runConfig"

interface LaunchBarProps {
  submitting: boolean
  submitError: string | null
  errorCount: number
}

export function LaunchBar({ submitting, submitError, errorCount }: LaunchBarProps) {
  const values = useWatch() as RunFormInput
  const name = useController<RunFormInput, "name">({ name: "name" })

  return (
    <div className="sticky bottom-0 border-t bg-background/95 backdrop-blur">
      <div className="mx-auto flex w-full max-w-5xl flex-wrap items-center gap-x-4 gap-y-2 px-6 py-3">
        <p className="min-w-0 flex-1 basis-64 truncate text-[13px]">
          <span className="text-muted-foreground">
            {experimentSentence(values)}
          </span>
        </p>

        <Input
          {...name.field}
          placeholder="Run name (optional)"
          className="h-8 w-56 text-xs"
        />

        <CliPopover values={values} />

        <div className="flex items-center gap-3">
          {errorCount > 0 && (
            <span className="text-xs text-status-err">
              {errorCount} issue{errorCount > 1 ? "s" : ""}
            </span>
          )}
          {submitError && (
            <span className="max-w-64 truncate text-xs text-status-err" title={submitError}>
              {submitError}
            </span>
          )}
          <Button type="submit" size="sm" className="gap-1.5" disabled={submitting}>
            <Play className="size-3.5" />
            {submitting ? "Starting…" : "Start evaluation"}
          </Button>
        </div>
      </div>
    </div>
  )
}

function CliPopover({ values }: { values: RunFormInput }) {
  const [copied, setCopied] = useState(false)
  const command = toCliCommand(values)

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="gap-1.5 text-muted-foreground"
        >
          <Terminal className="size-3.5" /> CLI
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[440px] p-0">
        <div className="flex items-center justify-between border-b px-3 py-2">
          <span className="section-label">CLI equivalent</span>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-6"
            aria-label="Copy command"
            onClick={async () => {
              await navigator.clipboard.writeText(
                command.replace(/ \\\n {2}/g, " ")
              )
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
        </div>
        <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap break-all px-3 py-2 font-mono text-[11px] leading-relaxed text-muted-foreground">
          {command}
        </pre>
      </PopoverContent>
    </Popover>
  )
}
