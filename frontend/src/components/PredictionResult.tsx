import React from 'react';
import type { PredictionData } from '../types/api';
import { formatPrice, formatPercent } from '../utils/formatters';

export default function PredictionResult({ data }: { data: PredictionData }) {
  return (
    <div className="p-4 border rounded-md">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-lg font-semibold">{data.ticker}</div>
          <div className="text-sm text-gray-500">{formatPrice(data.current_price)} → {formatPrice(data.predicted_price)}</div>
        </div>
        <div className={`px-2 py-1 rounded ${data.change_direction === 'up' ? 'bg-green-100 text-green-700' : data.change_direction === 'down' ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-700'}`}>
          {data.change_direction === 'up' ? '▲' : data.change_direction === 'down' ? '▼' : '—'} {formatPercent(data.change_percent)}
        </div>
      </div>
      <div className="mt-2 text-sm text-gray-600">Confidence: {data.confidence}</div>
    </div>
  );
}
