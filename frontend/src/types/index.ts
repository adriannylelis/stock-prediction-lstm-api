export interface PredictionResponse {
  success: boolean;
  data: PredictionData;
}

export interface HistoricalDataPoint {
  date: string;
  price: number;
}

export interface PredictionData {
  ticker: string;
  predicted_price: number;
  current_price: number;
  change_percent: number;
  change_direction: 'alta' | 'baixa' | 'neutra';
  prediction_date: string;
  confidence?: 'alta' | 'média' | 'baixa';
  timestamp: string;
  historical_data?: HistoricalDataPoint[];
}

export interface Stock {
  symbol: string;
  name: string;
  market?: string;
}

export interface StocksResponse {
  success: boolean;
  data: Stock[];
  count: number;
}

export interface ApiError {
  error: string;
  message: string;
  status: number;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  service: string;
}

export interface ModelInfo {
  model_type: string;
  architecture: string;
  input_size: number;
  hidden_size: number;
  num_layers: number;
  dropout: number;
  lookback_days: number;
  training: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    optimizer: string;
    loss_function: string;
  };
  performance: {
    test_mae: number;
    test_rmse: number;
    test_mape: number;
    test_r2: number;
    directional_accuracy: number;
  };
}
