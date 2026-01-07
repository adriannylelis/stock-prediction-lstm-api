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
    <Card className="transition-all duration-300 hover:shadow-lg hover:scale-[1.02]">
      <CardHeader>
        <CardTitle className="text-lg sm:text-xl">Price Prediction Chart</CardTitle>
      </CardHeader>
      <CardContent className="p-2 sm:p-6">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis 
              dataKey="date" 
              className="text-xs"
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              domain={[minPrice - padding, maxPrice + padding]}
              className="text-xs"
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => `$${value.toFixed(2)}`}
            />
            <Tooltip 
              formatter={(value: number | undefined) => value !== undefined ? [`$${value.toFixed(2)}`, ''] : ['', '']}
              labelStyle={{ color: '#000' }}
              contentStyle={{ 
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: '1px solid #ccc',
                borderRadius: '8px',
                fontSize: '14px',
              }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <ReferenceLine 
              y={currentPrice} 
              stroke="#94a3b8" 
              strokeDasharray="3 3" 
              label={{ value: 'Current', fontSize: 11 }}
            />
            <Line
              type="monotone"
              dataKey="current"
              stroke="#3b82f6"
              strokeWidth={3}
              name="Current Price"
              dot={{ r: 6 }}
              animationDuration={1000}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#10b981"
              strokeWidth={3}
              strokeDasharray="5 5"
              name="Predicted Price"
              dot={{ r: 6 }}
              animationDuration={1000}
              animationBegin={300}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
