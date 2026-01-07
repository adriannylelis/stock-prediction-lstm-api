import { useState } from 'react';
import { stockApi } from '@/services/api';
import type { PredictionData } from '@/types';

export function usePrediction() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = async (ticker: string) => {
    if (!ticker) {
      setError('Please select a stock ticker');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const data = await stockApi.predictStock(ticker);
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch prediction');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setPrediction(null);
    setError(null);
    setLoading(false);
  };

  return {
    prediction,
    loading,
    error,
    predict,
    reset,
  };
}
