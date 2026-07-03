import { useMutation, useQueryClient } from "@tanstack/react-query"

import { api } from "./client"
import type { RunStatus, RunSubmission } from "./types"

interface SubmitResponse {
  run_id: string
  status: RunStatus
  queue_position: number | null
}

export function useSubmitRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (submission: RunSubmission) =>
      api.post<SubmitResponse>("/api/runs", submission),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] })
    },
  })
}

export function useCancelRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (runId: string) =>
      api.post<{ run_id: string; status: RunStatus }>(
        `/api/runs/${runId}/cancel`
      ),
    onSuccess: (_data, runId) => {
      queryClient.invalidateQueries({ queryKey: ["run", runId] })
      queryClient.invalidateQueries({ queryKey: ["runs"] })
    },
  })
}

export function useDeleteRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (runId: string) => api.delete(`/api/runs/${runId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] })
    },
  })
}
