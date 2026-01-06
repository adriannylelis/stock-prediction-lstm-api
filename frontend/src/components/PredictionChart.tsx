import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface ChartDataPoint {
  date: string;
  current?: number;
  predicted?: number;
}

interface PredictionChartProps {
  currentPrice: number;
  predictedPrice: number;
  predictionDate: string;
}

export function PredictionChart({ 
  currentPrice, 
  predictedPrice,
  predictionDate 
}: PredictionChartProps) {
  const data: ChartDataPoint[] = [
    { 
      date: 'Today', 
      current: currentPrice,
      predicted: currentPrice,
    },
    { 
      date: new Date(predictionDate).toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
      }),
      predicted: predictedPrice,
    },
  ];

  const minPrice = Math.min(currentPrice, predictedPrice);
  const maxPrice = Math.max(currentPrice, predictedPrice);
  const padding = (maxPrice - minPrice) * 0.2 || 1;

  return (
    <Card className="transition-all duration-300 hover:shadow-lg">
      <CardHeader>
        <CardTitle>Price Prediction Chart</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis 
              dataKey="date" 
              className="text-xs"
            />
            <YAxis 
              domain={[minPrice - padding, maxPrice + padding]}
              className="text-xs"
              tickFormatter={(value) => `$${value.toFixed(2)}`}
            />
            <Tooltip 
              formatter={(value: number) => [`$${value.toFixed(2)}`, '']}
              labelStyle={{ color: '#000' }}
              contentStyle={{ 
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: '1px solid #ccc',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <ReferenceLine 
              y={currentPrice} 
              stroke="#94a3b8" 
              strokeDasharray="3 3" 
              label="Current"
            />
            <Line
              type="monotone"
              dataKey="current"
              stroke="#3b82f6"
              strokeWidth={3}
              name="Current Price"
              dot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#10b981"
              strokeWidth={3}
              strokeDasharray="5 5"
              name="Predicted Price"
              dot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
