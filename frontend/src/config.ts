export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001';

export const API_ENDPOINTS = {
  HEALTH: '/health',
  MODEL_INFO: '/model/info',
  PREDICT: '/predict',
} as const;
