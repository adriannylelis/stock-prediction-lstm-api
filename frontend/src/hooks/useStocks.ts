import { useMemo } from 'react';
import type { Stock } from '@/types';

// Fonte única de verdade para o front: apenas PETR4.SA suportado no backend atual.
export function useStocks() {
  const stocks = useMemo<Stock[]>(
    () => [
      {
        symbol: 'PETR4.SA',
        name: 'Petrobras PN',
        market: 'B3',
      },
    ],
    []
  );

  return {
    stocks,
    loading: false,
    error: null as string | null,
  };
}
