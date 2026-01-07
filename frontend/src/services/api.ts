import axios, { AxiosError } from 'axios';
import type { PredictionResponse, HealthResponse, ModelInfo, ApiError, PredictionData } from '@/types';

const API_BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) || 'http://localhost:5001';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

api.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiError>) => {
    if (error.response?.status === 404) {
      throw new Error(error.response.data?.message || 'Stock ticker not found');
    }
    if (error.response?.status === 400) {
      throw new Error(error.response.data?.message || 'Invalid request');
    }
    if (error.response?.status === 503) {
      throw new Error('Service temporarily unavailable. Please try again later.');
    }
    if (error.response?.status === 500) {
      throw new Error('Internal server error. Please try again.');
    }
    throw new Error(error.message || 'An unexpected error occurred');
  }
);

export const stockApi = {
  healthCheck: async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  },

  getModelInfo: async (): Promise<ModelInfo> => {
    const response = await api.get<ModelInfo>('/model/info');
    return response.data;
  },

  predictStock: async (ticker: string): Promise<PredictionData> => {
    const response = await api.post<PredictionResponse>('/predict', { ticker });
    return response.data.data;
  },
};
