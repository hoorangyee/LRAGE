import { useEffect, useRef, useState } from "react"
import { useQueryClient } from "@tanstack/react-query"

import type { ProgressSnapshot, RunDetail, RunStatus } from "./types"

export interface RunLogLine {
  id: number
  level: string
  message: string
}

const LOG_CAP = 1000

/**
 * Subscribe to a run's SSE stream. Events only ever write into the TanStack
 * Query cache (status/phase/progress) or the local log buffer — components
 * read run state via useRun, never from the EventSource directly. EventSource
 * auto-reconnects and sends Last-Event-ID, so the server replays what we
 * missed; the useRun polling fallback covers everything else.
 */
export function useRunStream(runId: string | undefined, enabled: boolean) {
  const queryClient = useQueryClient()
  const [logs, setLogs] = useState<RunLogLine[]>([])
  const [connected, setConnected] = useState(false)
  const seenLogIds = useRef<Set<number>>(new Set())

  useEffect(() => {
    if (!runId || !enabled) return

    setLogs([])
    seenLogIds.current = new Set()
    const es = new EventSource(`/api/runs/${runId}/events`)

    const patchRun = (patch: (old: RunDetail) => RunDetail) => {
      queryClient.setQueryData<RunDetail>(["run", runId], (old) =>
        old ? patch(old) : old
      )
    }
    const patchProgress = (progress: Partial<ProgressSnapshot>) => {
      patchRun((old) => ({
        ...old,
        progress: {
          phase: null,
          desc: null,
          n: null,
          total: null,
          pct: null,
          ...old.progress,
          ...progress,
        },
      }))
    }

    es.onopen = () => setConnected(true)
    es.onerror = () => setConnected(false)

    es.addEventListener("status", (e) => {
      const data = JSON.parse(e.data) as { status: RunStatus }
      patchRun((old) => ({ ...old, status: data.status }))
    })
    es.addEventListener("phase", (e) => {
      const data = JSON.parse(e.data) as { phase: string }
      patchProgress({ phase: data.phase, desc: null, n: null, total: null, pct: null })
    })
    es.addEventListener("progress", (e) => {
      const data = JSON.parse(e.data)
      patchProgress({
        desc: data.desc,
        n: data.n,
        total: data.total,
        pct: data.pct,
      })
    })
    es.addEventListener("log", (e) => {
      const data = JSON.parse(e.data) as {
        id: number
        level: string
        message: string
      }
      // Last-Event-ID replay can resend lines we already have.
      if (seenLogIds.current.has(data.id)) return
      seenLogIds.current.add(data.id)
      setLogs((prev) =>
        [...prev, { id: data.id, level: data.level, message: data.message }].slice(
          -LOG_CAP
        )
      )
    })
    es.addEventListener("done", () => {
      es.close()
      setConnected(false)
      queryClient.invalidateQueries({ queryKey: ["run", runId] })
      queryClient.invalidateQueries({ queryKey: ["runs"] })
      queryClient.invalidateQueries({ queryKey: ["results", runId] })
    })

    return () => {
      es.close()
      setConnected(false)
    }
  }, [runId, enabled, queryClient])

  return { logs, connected }
}
