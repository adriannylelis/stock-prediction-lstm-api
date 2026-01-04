import React from 'react';
import type { ModelInfo } from '../types/api';

export default function ModelInfoModal({ isOpen, onClose, modelInfo }: { isOpen: boolean; onClose: () => void; modelInfo?: ModelInfo }) {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center">
      <div className="bg-white p-4 rounded-md max-w-lg w-full">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Model Info</h3>
          <button onClick={onClose} className="text-sm text-gray-500">Close</button>
        </div>
        {modelInfo ? (
          <div className="mt-3 text-sm text-gray-700">
            <div><strong>Architecture:</strong> {modelInfo.architecture}</div>
            <div><strong>Input size:</strong> {modelInfo.input_size}</div>
            <div><strong>Hidden size:</strong> {modelInfo.hidden_size}</div>
            <div><strong>Lookback:</strong> {modelInfo.lookback}</div>
          </div>
        ) : (
          <div className="mt-3 text-sm text-gray-500">No model info available</div>
        )}
      </div>
    </div>
  );
}
