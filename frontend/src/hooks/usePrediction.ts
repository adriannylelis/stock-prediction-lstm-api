import { useState } from 'react';
import { stockApi } from '@/services/api';
import type { PredictionData } from '@/types';

const mapConfidenceToPt = (value: string | undefined | null): 'alta' | 'média' | 'baixa' => {
  if (!value) return 'média'; // Default to medium if undefined/null
  const v = value.toLowerCase();
  if (v === 'high' || v === 'alta') return 'alta';
  if (v === 'medium' || v === 'média' || v === 'media') return 'média';
  return 'baixa';
};

export function usePrediction() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const predict = async (ticker: string, includeHistory: boolean = true) => {
    if (!ticker) {
      setError('Please select a stock ticker');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const data = await stockApi.predictStock(ticker, includeHistory);
      setPrediction({
        ...data,
        confidence: data.confidence ? mapConfidenceToPt(data.confidence) : 'média',
      });
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
