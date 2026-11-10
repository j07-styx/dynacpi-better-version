/** Real, local DynaCPI v5 benchmark adapter. No arbitrary molecule inference is offered. */
export type BackendHealth = {
  status: string;
  service: string;
  model: string;
  active_variant: string;
  checkpoint_count: number;
  full_pairs: number;
  training_proteins: number;
  validation_proteins: number;
  heldout_proteins: number;
};
export type Score = {
  macro_rmse: number;
  pooled_rmse: number;
  centered_mse: number;
  protein_bias_mse: number;
  n_proteins?: number;
};
export type MetricsResponse = {
  status: string;
  selected: string;
  test_chemical_only: Score;
  test_selected: Score;
  relative_rmse_reduction_pct: number;
  n_better: number;
  n_worse: number;
  validation_variants: Record<string, Score>;
  limitations: string[];
};
export type TrainingPoint = {
  epoch: number;
  standardized_training_objective: number;
  validation_macro_rmse: number;
  validation_centered_mse: number;
  validation_protein_bias_mse: number;
};
export type TrainingResponse = {
  variant: string;
  runs: {
    seed: number;
    variant: string;
    best_epoch: number;
    best_val: number;
    history: TrainingPoint[];
  }[];
};
export type BenchmarkExample = { demo_id: string; label: string };
export type ExamplesResponse = { examples: BenchmarkExample[]; selection: string };
export type BenchmarkPrediction = {
  demo_id: string;
  prediction_pKd: number;
  reference_pKd: number;
  absolute_error: number;
  seed_prediction_std: number;
  heldout: string;
  benchmark: string;
  label: string;
};
export class ApiError extends Error {
  constructor(message: string, public kind: "network" | "http" | "invalid_response" | "disconnected", public status?: number) {
    super(message);
    this.name = "ApiError";
  }
}

/** Local Vite proxy /api/v1 -> localhost:8782. An explicit env overrides this.
 * Lovable's HTTPS-hosted preview MUST NOT contact the viewer's localhost. */
export function getApiUrl(): string | null {
  const configured = (import.meta.env.VITE_DYNACPI_API_URL as string | undefined)?.trim();
  if (configured) return configured.replace(/\/+$/, "");
  if (typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"))
    return "/api/v1";
  return null;
}

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const root = getApiUrl();
  if (!root) throw new ApiError("Local v5 backend is not configured for this preview. Run the local frontend and backend on your PC.", "disconnected");
  let response: Response;
  try {
    response = await fetch(root + path, {
      ...options,
      headers: { Accept: "application/json", ...(options?.body ? { "Content-Type": "application/json" } : {}), ...options?.headers },
    });
  } catch {
    throw new ApiError("Cannot reach the local DynaCPI v5 backend. Confirm server port 8782 and frontend proxy.", "network");
  }
  let obj: unknown;
  try { obj = await response.json(); } catch { throw new ApiError("Backend returned non-JSON content.", "invalid_response", response.status); }
  if (!response.ok) {
    const message = obj && typeof obj === "object" && "error" in obj ? String((obj as {error: unknown}).error) : response.statusText;
    throw new ApiError("Backend returned " + response.status + ": " + message, "http", response.status);
  }
  return obj as T;
}
export const checkHealth = async (): Promise<boolean> => {
  try {
    const info = await api<BackendHealth>("/health");
    return info.status === "ok" && info.service === "dynacpi-full-davis-v5" && info.checkpoint_count === 2;
  } catch { return false; }
};
export const fetchHealth = (): Promise<BackendHealth> => api<BackendHealth>("/health");
export const fetchMetrics = (): Promise<MetricsResponse> => api<MetricsResponse>("/metrics");
export const fetchTraining = (): Promise<TrainingResponse> => api<TrainingResponse>("/training");
export const fetchExamples = (): Promise<ExamplesResponse> => api<ExamplesResponse>("/examples");
export async function runBenchmark(demoId: string): Promise<BenchmarkPrediction> {
  if (!/^DV-(0[1-9]|1[0-2])$/.test(demoId)) throw new ApiError("Only the backend's twelve fixed benchmark examples are supported.", "invalid_response");
  const o = await api<BenchmarkPrediction>("/predict", {
    method: "POST",
    body: JSON.stringify({ demo_id: demoId }),
  });
  if (!o || !["prediction_pKd","reference_pKd","absolute_error","seed_prediction_std"].every(k => Number.isFinite((o as unknown as Record<string, unknown>)[k])))
    throw new ApiError("Benchmark inference response is missing numeric model outputs.", "invalid_response");
  return o;
}

/** Kept for the original skeleton's unit tests; v5 uses runBenchmark instead. */
export function parsePrediction(json: unknown): { affinity: number; unit?: string; model?: string } {
  if (!json || typeof json !== "object") throw new ApiError("Response is not a JSON object.", "invalid_response");
  const o = json as Record<string, unknown>;
  const cand = o["affinity"] ?? o["prediction"] ?? o["predicted_affinity"] ?? o["value"];
  const n = typeof cand === "string" ? Number(cand) : cand;
  if (typeof n !== "number" || !Number.isFinite(n))
    throw new ApiError("Response has no numeric 'affinity' field.", "invalid_response");
  const result: { affinity: number; unit?: string; model?: string } = { affinity: n };
  const unit = typeof o["unit"] === "string" ? o["unit"] : o["units"];
  if (typeof unit === "string") result.unit = unit;
  if (typeof o["model"] === "string") result.model = o["model"];
  return result;
}


/** Final all-DAVIS refit: separate API and separate scientific status from v5 evaluation.
 * /api/v2 uses a Vite same-origin proxy to 127.0.0.1:8783 on the owner's PC.
 * On a hosted preview we intentionally do not call the visitor's localhost. */
export type V6Health = {
  status: string;
  service: "dynacpi-v6-all-davis";
  n_proteins_trained: number;
  n_compounds_trained: number;
  n_pairs_trained: number;
  n_checkpoints: number;
  supports_arbitrary_smiles: boolean;
  supports_novel_protein_sequences: boolean;
  no_independent_test_for_final_refit: boolean;
};
export type V6Prediction = {
  status: string;
  model: string;
  model_version: string;
  prediction_pKd: number;
  unit: "pKd";
  seed_prediction_std: number;
  protein_in_original_davis: boolean;
  compound_in_original_davis: boolean;
  known_training_pair: boolean;
  experimental_extrapolation: boolean;
  chemical_prior_source: string;
  protein_embedding_source: string;
  comparison_scope: string;
  notice: string;
  n_training_pairs: number;
};
export type V6Example = { smiles: string; sequence: string; label: string; note: string };
export type V6FitMetrics = {
  status: "VERIFIED_ALL_DATA_TRAINING_FIT_NOT_HELDOUT";
  all_data_fit_baseline_chemical_mean: Score;
  all_data_fit_v6_ensemble: Score;
  all_data_fit_improvement_relative_baseline_percent: number;
  pairs_train: number;
  proteins_train_entries: number;
  unique_protein_sequences: number;
  compounds_train: number;
  strict_warning: string;
  training_runs: { seed: number; epochs: number; history: { epoch: number; standardized_training_objective: number }[] }[];
  v5_separate_heldout_evaluation: {
    tested_earlier_model_not_v6_final_refit: boolean;
    macro_rmse_v5_heldout: number;
    macro_rmse_chemical_baseline_v5_heldout: number;
    heldout_proteins: number;
  };
};
export function getV6ApiUrl(): string | null {
  const explicit = (import.meta.env["VITE_DYNACPI_V6_API_URL"] as string | undefined)?.trim();
  if (explicit) return explicit.replace(/\/+$/, "");
  if (typeof window !== "undefined" &&
      (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"))
    return "/api/v2";
  return null;
}
async function apiV6<T>(path: string, options?: RequestInit): Promise<T> {
  const root = getV6ApiUrl();
  if (!root) throw new ApiError("v6 inference is only available on the local research workstation.", "disconnected");
  let response: Response;
  try {
    response = await fetch(root+path, {
      ...options,
      headers: { Accept:"application/json",
        ...(options?.body ? {"Content-Type":"application/json"} : {}),
        ...options?.headers },
    });
  } catch (e) {
    if (e instanceof Error && e.name === "AbortError") throw e;
    throw new ApiError("Cannot reach the v6 model on port 8783. Check the local backend and restart if needed.", "network");
  }
  let raw: unknown;
  try { raw = await response.json(); }
  catch { throw new ApiError("v6 returned unreadable JSON.", "invalid_response", response.status); }
  if (!response.ok) {
    let detail = response.statusText;
    if (raw && typeof raw === "object" && "error" in raw) detail = String((raw as {error:unknown}).error);
    throw new ApiError("v6 API returned "+response.status+": "+detail, "http", response.status);
  }
  return raw as T;
}
export const fetchV6Health = () => apiV6<V6Health>("/health");
export const fetchV6Example = () => apiV6<V6Example>("/example");
export const fetchV6FitMetrics = () => apiV6<V6FitMetrics>("/metrics");

/** All user input is sent to the actual v6 checkpoint-backed model.
 * Only backend-derived prediction values are shown. */
export async function predictV6(input: {smiles:string;sequence:string}, signal?:AbortSignal):Promise<V6Prediction> {
  const result = await apiV6<V6Prediction>("/predict",{
    method:"POST",
    body:JSON.stringify({smiles:input.smiles,sequence:input.sequence}),
    ...(signal ? {signal} : {})
  });
  if (!result || result.status !== "ok" ||
      !Number.isFinite(result.prediction_pKd) ||
      !Number.isFinite(result.seed_prediction_std) ||
      typeof result.notice !== "string" ||
      typeof result.protein_in_original_davis !== "boolean" ||
      typeof result.compound_in_original_davis !== "boolean") {
    throw new ApiError("v6 inference response is incomplete or malformed.", "invalid_response");
  }
  return result;
}
