import { useQuery } from "@tanstack/react-query"

import { api } from "./client"
import type {
  HealthResponse,
  PresetsResponse,
  RegistriesResponse,
  ResultsResponse,
  RunDetail,
  RunListItem,
  SampleDetailResponse,
  SamplePage,
  TaskMeta,
} from "./types"

export function useCompare(runIds: string[]) {
  return useQuery({
    queryKey: ["compare", runIds.join(",")],
    queryFn: () =>
      api.get<import("./types").CompareResponse>(
        `/api/compare?run_ids=${runIds.join(",")}`
      ),
    enabled: runIds.length >= 2,
    staleTime: Infinity,
  })
}

export interface SampleFilters {
  task?: string
  page?: number
  q?: string
  incorrectOnly?: boolean
  judgeMin?: number
  judgeMax?: number
}

export function useSamples(runId: string | undefined, filters: SampleFilters) {
  const params = new URLSearchParams()
  if (filters.task) params.set("task", filters.task)
  if (filters.page) params.set("page", String(filters.page))
  if (filters.q) params.set("q", filters.q)
  if (filters.incorrectOnly) params.set("incorrect_only", "true")
  if (filters.judgeMin != null) params.set("judge_min", String(filters.judgeMin))
  if (filters.judgeMax != null) params.set("judge_max", String(filters.judgeMax))
  const qs = params.toString()
  return useQuery({
    queryKey: ["samples", runId, qs],
    queryFn: () =>
      api.get<SamplePage>(`/api/runs/${runId}/samples${qs ? `?${qs}` : ""}`),
    enabled: !!runId,
    staleTime: Infinity,
    placeholderData: (prev) => prev,
  })
}

export function useSample(
  runId: string | undefined,
  task: string | null | undefined,
  docId: number | null
) {
  return useQuery({
    queryKey: ["sample", runId, task, docId],
    queryFn: () =>
      api.get<SampleDetailResponse>(
        `/api/runs/${runId}/samples/${task}/${docId}`
      ),
    enabled: !!runId && !!task && docId != null,
    staleTime: Infinity,
  })
}

const ACTIVE_STATUSES = new Set(["queued", "running", "cancelling"])

export function useRuns() {
  return useQuery({
    queryKey: ["runs"],
    queryFn: () => api.get<{ runs: RunListItem[] }>("/api/runs"),
    select: (data) => data.runs,
    refetchInterval: (query) =>
      query.state.data?.runs.some((r) => ACTIVE_STATUSES.has(r.status))
        ? 5000
        : false,
  })
}

export function useRun(runId: string | undefined, opts?: { poll?: boolean }) {
  return useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.get<RunDetail>(`/api/runs/${runId}`),
    enabled: !!runId,
    refetchInterval: (query) => {
      if (!opts?.poll) return false
      const status = query.state.data?.status
      return status && ACTIVE_STATUSES.has(status) ? 5000 : false
    },
  })
}

export function useResults(runId: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ["results", runId],
    queryFn: () => api.get<ResultsResponse>(`/api/runs/${runId}/results`),
    enabled: !!runId && enabled,
    staleTime: Infinity, // completed runs never change
  })
}

export function useRunLogs(runId: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ["logs", runId],
    queryFn: async () => {
      const res = await fetch(`/api/runs/${runId}/logs`)
      if (!res.ok) throw new Error("Could not load logs")
      return res.text()
    },
    enabled: !!runId && enabled,
  })
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => api.get<HealthResponse>("/api/health"),
  })
}

export function useTasks() {
  return useQuery({
    queryKey: ["meta", "tasks"],
    queryFn: () => api.get<{ tasks: TaskMeta[] }>("/api/meta/tasks"),
    select: (data) => data.tasks,
    staleTime: Infinity,
    // First call indexes ~2.5k task YAMLs server-side and can take a while.
    retry: 2,
  })
}

export function useRegistries() {
  return useQuery({
    queryKey: ["meta", "registries"],
    queryFn: () => api.get<RegistriesResponse>("/api/meta/registries"),
    staleTime: Infinity,
  })
}

export function usePresets() {
  return useQuery({
    queryKey: ["meta", "presets"],
    queryFn: () => api.get<PresetsResponse>("/api/meta/presets"),
    staleTime: Infinity,
  })
}

export function useDevices() {
  return useQuery({
    queryKey: ["meta", "devices"],
    queryFn: () => api.get<{ devices: string[] }>("/api/meta/devices"),
    select: (data) => data.devices,
    staleTime: Infinity,
  })
}
