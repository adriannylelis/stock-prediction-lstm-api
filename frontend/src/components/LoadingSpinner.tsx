import React from 'react';

export default function LoadingSpinner({ text }: { text?: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-5 h-5 border-2 border-gray-300 border-t-transparent rounded-full animate-spin" />
      {text && <span className="text-sm text-gray-600">{text}</span>}
    </div>
  );
}
