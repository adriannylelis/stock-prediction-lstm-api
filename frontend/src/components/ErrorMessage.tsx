import React from 'react';

export default function ErrorMessage({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="p-4 bg-red-50 text-red-700 rounded-md">
      <div className="flex justify-between items-center">
        <div>{message}</div>
        {onRetry && (
          <button onClick={onRetry} className="ml-4 text-sm underline">
            Retry
          </button>
        )}
      </div>
    </div>
  );
}
