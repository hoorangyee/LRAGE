// DTOs mirroring lrage/webapp responses. Keep field names snake_case to
// match the API payloads exactly.

export interface TaskMeta {
  name: string
  type: string // "task" | "group" | "python_task"
}

export interface ModelType {
  key: string
  label: string
}

export interface RegistriesResponse {
  model_types: ModelType[]
  retrievers: string[]
  retriever_types: string[]
  rerankers: string[]
  reranker_types: string[]
}

export interface ModelPreset {
  label: string
  args: string
}

export interface PresetsResponse {
  model_types: ModelType[]
  model_presets: Record<string, ModelPreset[]>
  retriever_presets: Record<string, string[]>
  reranker_presets: string[]
}

export interface HealthResponse {
  status: string
  version: string
  tasks_loaded: boolean
}

export type RunStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled"

export interface GenKwargs {
  max_gen_toks?: number
  temperature?: number
  do_sample?: boolean
}

export interface RetrievalConfig {
  retrieve_docs: boolean
  top_k: number
  retriever?: string | null
  retriever_args?: string | null
  rerank: boolean
  reranker?: string | null
  reranker_args?: string | null
}

export interface RunConfig {
  model: string
  model_args: string
  tasks: string[]
  judge_model?: string | null
  judge_model_args?: string | null
  judge_device?: string | null
  judge_gen_kwargs?: GenKwargs | null
  num_fewshot?: number | null
  fewshot_as_multiturn: boolean
  batch_size: number | "auto"
  max_batch_size?: number | null
  device?: string | null
  limit?: number | null
  use_cache?: string | null
  cache_requests: boolean
  gen_kwargs?: GenKwargs | null
  system_instruction?: string | null
  apply_chat_template: boolean
  retrieval: RetrievalConfig
  log_samples: boolean
  predict_only: boolean
  random_seed: number
  numpy_random_seed: number
  torch_random_seed: number
  fewshot_random_seed: number
}

export interface ApiKeys {
  openai_api_key?: string
  hf_token?: string
}

export interface RunSubmission {
  name?: string
  config: RunConfig
  api_keys?: ApiKeys
}

export interface ProgressSnapshot {
  phase: string | null
  desc: string | null
  n: number | null
  total: number | null
  pct: number | null
}

export interface RunListItem {
  run_id: string
  name: string | null
  status: RunStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  model: string
  tasks: string[]
  error: string | null
  headline_metric?: { metric: string; value: number } | null
}

export interface RunDetail extends RunListItem {
  config: RunConfig
  progress: ProgressSnapshot | null
  queue_position: number | null
}

export interface MetricRow {
  task: string
  n_shot: string
  metric: string
  value: number
  stderr: number | null
}

export interface ResultsResponse {
  run_id: string
  table: MetricRow[]
  results: Record<string, unknown>
}

export interface SampleListItem {
  doc_id: number
  input: string
  target: string
  resp: string
  metrics: Record<string, number>
  rating: number | null
}

export interface SamplePage {
  tasks: string[]
  task: string | null
  total: number
  page: number
  page_size?: number
  items: SampleListItem[]
}

export interface SampleRecord {
  doc_id: number
  doc: unknown
  target: unknown
  arguments: unknown
  resps: unknown
  filtered_resps: unknown
  Rating?: number
  Explanation?: string
  [key: string]: unknown
}

export interface SampleDetailResponse {
  run_id: string
  task: string
  sample: SampleRecord
  metrics: Record<string, number>
}

export interface CompareRunInfo {
  run_id: string
  name: string | null
  model: string
  tasks: string[]
  config: RunConfig
  finished_at: string | null
}

export interface CompareRow {
  task: string
  metric: string
  values: Record<string, { value: number; stderr: number | null }>
}

export interface CompareResponse {
  runs: CompareRunInfo[]
  table: CompareRow[]
}

export type RunEvent =
  | { type: "status"; status: RunStatus }
  | { type: "phase"; phase: string }
  | {
      type: "progress"
      phase: string | null
      desc: string | null
      n: number
      total: number | null
      pct: number | null
    }
  | { type: "log"; level: string; message: string }
  | { type: "done"; status: RunStatus }
  | { type: "error"; message: string }
