import { useState } from 'react';
import { StockSelector } from './components/StockSelector';
import { PredictionChart } from './components/PredictionChart';
import { PredictionCard } from './components/PredictionCard';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { usePrediction } from './hooks/usePrediction';
import { TrendingUp } from 'lucide-react';
import type { Stock } from './types';

const AVAILABLE_STOCKS: Stock[] = [
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'MSFT', name: 'Microsoft Corporation' },
  { symbol: 'GOOGL', name: 'Alphabet Inc. (Google)' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.' },
  { symbol: 'TSLA', name: 'Tesla Inc.' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation' },
  { symbol: 'META', name: 'Meta Platforms Inc.' },
  { symbol: 'PETR4.SA', name: 'Petrobras (B3)' },
  { symbol: 'VALE3.SA', name: 'Vale S.A. (B3)' },
  { symbol: 'ITUB4.SA', name: 'Itaú Unibanco (B3)' },
];

function App() {
  const [selectedStock, setSelectedStock] = useState('');
  const { prediction, loading, error, predict } = usePrediction();

  const handlePredict = () => {
    if (selectedStock) {
      predict(selectedStock);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 transition-colors duration-300">
      <div className="container mx-auto px-4 py-6 sm:py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-6 sm:mb-8">
          <div className="flex items-center justify-center gap-2 sm:gap-3 mb-4">
            <TrendingUp className="w-8 h-8 sm:w-10 sm:h-10 text-primary animate-pulse" />
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 to-green-600 bg-clip-text text-transparent">
              Stock Prediction Dashboard
            </h1>
          </div>
          <p className="text-sm sm:text-base md:text-lg text-muted-foreground px-4">
            AI-powered LSTM predictions for next-day stock prices
          </p>
        </div>

        {/* Selection Panel */}
        <Card className="mb-6 sm:mb-8 shadow-lg transition-all duration-300 hover:shadow-xl">
          <CardHeader>
            <CardTitle className="text-lg sm:text-xl">Select Stock & Predict</CardTitle>
            <CardDescription className="text-xs sm:text-sm">
              Choose a stock ticker and get AI-powered price predictions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1">
                <label className="text-xs sm:text-sm font-medium mb-2 block">
                  Stock Ticker
                </label>
                <StockSelector
                  stocks={AVAILABLE_STOCKS}
                  value={selectedStock}
                  onChange={setSelectedStock}
                  disabled={loading}
                />
              </div>
              <div className="flex items-end">
                <Button
                  onClick={handlePredict}
                  disabled={!selectedStock || loading}
                  size="lg"
                  className="w-full md:w-auto min-w-[150px] transition-all duration-200 hover:scale-105"
                >
                  {loading ? (
                    <>
                      <span className="animate-spin mr-2">⏳</span>
                      Predicting...
                    </>
                  ) : (
                    'Get Prediction'
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Error State */}
        {error && !loading && (
          <div className="mb-6 sm:mb-8 animate-in fade-in-50 slide-in-from-top-2">
            <ErrorMessage message={error} />
          </div>
        )}

        {/* Loading State */}
        {loading && <LoadingSpinner />}

        {/* Success State - Prediction Results */}
        {prediction && !loading && !error && (
          <div className="animate-in fade-in-50 slide-in-from-bottom-4 duration-700">
            <div className="grid md:grid-cols-2 gap-4 sm:gap-6">
              <PredictionCard prediction={prediction} />
              <PredictionChart
                currentPrice={prediction.current_price}
                predictedPrice={prediction.predicted_price}
                predictionDate={prediction.prediction_date}
              />
            </div>

            {/* Additional Info */}
            <Card className="mt-4 sm:mt-6 bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800 transition-all duration-300 hover:shadow-lg">
              <CardHeader>
                <CardTitle className="text-base sm:text-lg">ℹ️ About This Prediction</CardTitle>
              </CardHeader>
              <CardContent className="text-xs sm:text-sm text-muted-foreground">
                <ul className="space-y-2">
                  <li>• Predictions are generated using a trained LSTM neural network model</li>
                  <li>• Model uses 60 days of historical data with 14 technical indicators</li>
                  <li>• Confidence levels: <strong>High</strong> (&lt;2% change), <strong>Medium</strong> (2-5%), <strong>Low</strong> (&gt;5%)</li>
                  <li>• This is for educational purposes only - not financial advice</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Empty State */}
        {!prediction && !loading && !error && (
          <Card className="text-center py-12 sm:py-16 transition-all duration-300 hover:shadow-lg">
            <CardContent>
              <TrendingUp className="w-12 h-12 sm:w-16 sm:h-16 mx-auto mb-4 text-muted-foreground animate-bounce" />
              <h3 className="text-lg sm:text-xl font-semibold mb-2">Ready to Predict</h3>
              <p className="text-sm sm:text-base text-muted-foreground px-4">
                Select a stock ticker above and click "Get Prediction" to start
              </p>
            </CardContent>
          </Card>
        )}

        {/* Footer */}
        <div className="mt-8 sm:mt-12 text-center text-xs sm:text-sm text-muted-foreground">
          <p>Powered by LSTM Neural Networks • Built with React + TypeScript + shadcn/ui</p>
          <p className="mt-1">© 2026 Stock Prediction API</p>
        </div>
      </div>
    </div>
  );
}

export default App;
