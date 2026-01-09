import { useState, useEffect } from 'react';
import { stockApi } from '@/services/api';
import type { Stock } from '@/types';

export function useStocks() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStocks = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await stockApi.getStocks();
        setStocks(data);
      } catch (err) {
        console.error('Erro ao buscar ações:', err);
        setError(err instanceof Error ? err.message : 'Falha ao carregar ações');
        setStocks([]);
      } finally {
        setLoading(false);
      }
    };

    fetchStocks();
  }, []);

  return {
    stocks,
    loading,
    error,
  };
}
