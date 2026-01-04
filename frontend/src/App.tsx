import { useState } from 'react';
import './App.css';
import PredictionForm from './components/PredictionForm';
import PredictionResult from './components/PredictionResult';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import ModelInfoModal from './components/ModelInfoModal';
import { usePrediction } from './hooks/usePrediction';
import { fetchModelInfo } from './services/api';
import type { ModelInfo } from './types/api';

function App() {
  const { data, loading, error, predict, reset } = usePrediction();
  const [modelInfo, setModelInfo] = useState<ModelInfo | undefined>(undefined);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);

  async function openModel() {
    setLoadingModel(true);
    setModelError(null);
    try {
      const m = await fetchModelInfo();
      setModelInfo(m);
      setIsModalOpen(true);
    } catch (err: any) {
      setModelError(err?.message || 'Erro ao buscar info do modelo');
    } finally {
      setLoadingModel(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-2xl mx-auto">
        <header className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Stock Prediction</h1>
          <div>
            <button className="px-3 py-1 bg-gray-200 rounded" onClick={openModel}>
              {loadingModel ? 'Loading...' : 'Model Info'}
            </button>
          </div>
        </header>

        <main className="space-y-4">
          <PredictionForm onSubmit={predict} loading={loading} />

          {loading && <LoadingSpinner text="Predizendo..." />}

          {error && <ErrorMessage message={error} onRetry={() => { if (typeof window !== 'undefined') window.location.reload(); }} />}

          {data && <PredictionResult data={data} />}

          <div className="mt-3">
            <button className="text-sm text-gray-500 underline" onClick={() => reset()}>
              Reset
            </button>
          </div>

          {modelError && <ErrorMessage message={modelError} />}
        </main>

        <ModelInfoModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} modelInfo={modelInfo} />
      </div>
    </div>
  );
}

export default App;
