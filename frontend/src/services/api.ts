import axios from 'axios';
import type { HealthResponse, ModelInfo, PredictionResponse } from '../types/api';
import { API_BASE_URL, API_ENDPOINTS } from '../config';

const client = axios.create({ baseURL: API_BASE_URL, timeout: 5000 });

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await client.get(API_ENDPOINTS.HEALTH);
  return res.data as HealthResponse;
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await client.get(API_ENDPOINTS.MODEL_INFO);
  return res.data as ModelInfo;
}

export async function predictPrice(ticker: string): Promise<PredictionResponse> {
  const res = await client.post(API_ENDPOINTS.PREDICT, { ticker });
  return res.data as PredictionResponse;
}
