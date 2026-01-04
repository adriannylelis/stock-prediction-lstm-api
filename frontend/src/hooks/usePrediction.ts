import { useState } from 'react';
import type { PredictionData } from '../types/api';
import { predictPrice } from '../services/api';

interface UsePredictionReturn {
  data: PredictionData | null;
  loading: boolean;
  error: string | null;
  ticker: string;
  predict: (ticker: string) => Promise<void>;
  reset: () => void;
}

export function usePrediction(): UsePredictionReturn {
  const [data, setData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ticker, setTicker] = useState('');

  async function predict(t: string) {
    setLoading(true);
    setError(null);
    setTicker(t);
    try {
      const res = await predictPrice(t);
      setData(res.data);
    } catch (err: any) {
      setError(err?.message || 'Erro ao fazer predição');
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setData(null);
    setError(null);
    setTicker('');
  }

  return { data, loading, error, ticker, predict, reset };
}
