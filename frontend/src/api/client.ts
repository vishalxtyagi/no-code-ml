import axios from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

export interface ColumnProfile {
  dtype: string;
  kind: 'numerical' | 'categorical';
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  stats?: Record<string, number>;
  top_values?: Record<string, number>;
  outlier_count?: number;
}

export interface DatasetProfile {
  shape: { rows: number; columns: number };
  columns: Record<string, ColumnProfile>;
  missing_summary: Record<string, number>;
  duplicate_rows: number;
}

export interface UploadResponse {
  job_id: string;
  columns: string[];
  profile: DatasetProfile;
  detected_problem_type: 'classification' | 'regression';
  suggested_target: string;
  preprocessing_suggestions: string[];
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelScore {
  model: string;
  accuracy?: number;
  f1_weighted?: number;
  precision?: number;
  recall?: number;
  mae?: number;
  mse?: number;
  rmse?: number;
  r2?: number;
}

export interface PipelineResult {
  job_id: string;
  problem_type: 'classification' | 'regression';
  target_col: string;
  best_model: string;
  best_model_scores: Record<string, number>;
  comparison_table: ModelScore[];
  feature_importances: FeatureImportance[];
  explanation: string;
  preprocessing_suggestions: string[];
  dataset_shape: { rows: number; columns: number };
}

export const uploadDataset = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axios.post<UploadResponse>(`${BASE_URL}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const runPipeline = async (
  jobId: string,
  targetCol: string,
  tune: boolean = true
): Promise<PipelineResult> => {
  const response = await axios.post<PipelineResult>(
    `${BASE_URL}/run-pipeline/${jobId}`,
    { target_col: targetCol, tune }
  );
  return response.data;
};

export const getResults = async (jobId: string): Promise<PipelineResult> => {
  const response = await axios.get<PipelineResult>(`${BASE_URL}/results/${jobId}`);
  return response.data;
};

export const getDownloadUrl = (jobId: string): string =>
  `${BASE_URL}/download-model/${jobId}`;
