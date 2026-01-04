import React, { useState } from 'react';
import { validateTicker } from '../utils/validators';

export default function PredictionForm({ onSubmit, loading }: { onSubmit: (ticker: string) => void; loading: boolean }) {
  const [ticker, setTicker] = useState('');
  const [error, setError] = useState<string | null>(null);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!validateTicker(ticker)) {
      setError('Ticker inválido');
      return;
    }
    setError(null);
    onSubmit(ticker.trim().toUpperCase());
  }

  return (
    <form onSubmit={submit} className="flex gap-2 items-center">
      <input
        className={`px-3 py-2 border rounded-md ${error ? 'border-red-500' : 'border-gray-300'}`}
        placeholder="Ticker (e.g. AAPL)"
        value={ticker}
        onChange={(e) => setTicker(e.target.value)}
        disabled={loading}
      />
      <button className="px-3 py-2 bg-blue-600 text-white rounded-md" disabled={loading}>
        {loading ? 'Carregando...' : 'Prever'}
      </button>
      {error && <div className="text-red-500 text-sm">{error}</div>}
    </form>
  );
}
