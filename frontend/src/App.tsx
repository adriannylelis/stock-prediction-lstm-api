import { useState } from 'react';
import { StockSelector } from './components/StockSelector';
import { PredictionChart } from './components/PredictionChart';
import { PredictionCard } from './components/PredictionCard';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { usePrediction } from './hooks/usePrediction';
import { useStocks } from './hooks/useStocks';
import { TrendingUp } from 'lucide-react';

function App() {
  const [selectedStock, setSelectedStock] = useState('');
  const { prediction, loading, error, predict } = usePrediction();
  const { stocks, loading: loadingStocks, error: stocksError } = useStocks();

  const handlePredict = () => {
    if (selectedStock) {
      predict(selectedStock);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 transition-colors duration-300">
      <div className="container mx-auto px-4 py-6 sm:py-8 max-w-7xl">
        <div className="text-center mb-6 sm:mb-8">
          <div className="flex items-center justify-center gap-2 sm:gap-3 mb-4">
            <TrendingUp className="w-8 h-8 sm:w-10 sm:h-10 text-primary animate-pulse" />
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent">
              Painel de Previsão de Ações
            </h1>
          </div>
          <p className="text-sm sm:text-base md:text-lg text-muted-foreground px-4">
            Previsões de preços de ações com IA usando LSTM
          </p>
        </div>

        <Card className="mb-6 sm:mb-8 shadow-lg transition-all duration-300 hover:shadow-xl">
          <CardHeader>
            <CardTitle className="text-lg sm:text-xl">Selecione a Ação</CardTitle>
            <CardDescription className="text-xs sm:text-sm">
              Escolha um ticker e obtenha previsões de preço com IA
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1">
                <label className="text-xs sm:text-sm font-medium mb-2 block">
                  Ticker da Ação
                </label>
                {loadingStocks ? (
                  <div className="h-10 bg-muted animate-pulse rounded-md"></div>
                ) : stocksError ? (
                  <div className="text-xs text-red-500">Erro ao carregar ações</div>
                ) : (
                  <StockSelector
                    stocks={stocks}
                    value={selectedStock}
                    onChange={setSelectedStock}
                    disabled={loading}
                  />
                )}
              </div>
              <div className="flex items-end">
                <Button
                  onClick={handlePredict}
                  disabled={!selectedStock || loading || loadingStocks}
                  size="lg"
                  className="w-full md:w-auto min-w-[150px] transition-all duration-200 hover:scale-105"
                >
                  {loading ? (
                    <>
                      <span className="animate-spin mr-2">⏳</span>
                      Prevendo...
                    </>
                  ) : (
                    'Obter Previsão'
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {error && !loading && (
          <div className="mb-6 sm:mb-8 animate-in fade-in-50 slide-in-from-top-2">
            <ErrorMessage message={error} />
          </div>
        )}

        {loading && <LoadingSpinner />}


        {prediction && !loading && !error && (
          <div className="animate-in fade-in-50 slide-in-from-bottom-4 duration-700">
            <div className="grid md:grid-cols-2 gap-4 sm:gap-6">
              <PredictionCard prediction={prediction} />
              <PredictionChart
                currentPrice={prediction.current_price}
                predictedPrice={prediction.predicted_price}
                predictionDate={prediction.prediction_date}
                historicalData={prediction.historical_data}
              />
            </div>

            <Card className="mt-4 sm:mt-6 bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800 transition-all duration-300 hover:shadow-lg">
              <CardHeader>
                <CardTitle className="text-base sm:text-lg">ℹ️ Sobre Esta Previsão</CardTitle>
              </CardHeader>
              <CardContent className="text-xs sm:text-sm text-muted-foreground">
                <ul className="space-y-2">
                  <li>• Previsões geradas usando um modelo de rede neural LSTM treinado</li>
                  <li>• Modelo utiliza 60 dias de dados históricos com 14 indicadores técnicos</li>
                  <li>• Níveis de confiança: <strong>Alta</strong> (&lt;2% mudança), <strong>Média</strong> (2-5%), <strong>Baixa</strong> (&gt;5%)</li>
                  <li>• Apenas para fins educacionais - não constitui aconselhamento financeiro</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        )}

        {!prediction && !loading && !error && (
          <Card className="text-center py-12 sm:py-16 transition-all duration-300 hover:shadow-lg">
            <CardContent>
              <TrendingUp className="w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-4 text-muted-foreground animate-bounce" />
              <h3 className="text-lg sm:text-xl font-semibold mb-2">Pronto para Prever</h3>
              <p className="text-sm sm:text-base text-muted-foreground px-4">
                Selecione um ticker acima e clique em "Obter Previsão" para começar
              </p>
            </CardContent>
          </Card>
        )}

        <div className="mt-8 sm:mt-12 text-center text-xs sm:text-sm text-muted-foreground">
          <p>Desenvolvido com Redes Neurais LSTM • React + TypeScript + shadcn/ui</p>
          <p className="mt-1">© 2026 API de Previsão de Ações</p>
        </div>
      </div>
    </div>
  );
}

export default App;
