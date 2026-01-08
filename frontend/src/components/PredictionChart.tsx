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
import type { HistoricalDataPoint } from '@/types';

interface ChartDataPoint {
  date: string;
  historical?: number;
  current?: number;
  predicted?: number;
}

interface PredictionChartProps {
  currentPrice: number;
  predictedPrice: number;
  predictionDate: string;
  historicalData?: HistoricalDataPoint[];
}

export function PredictionChart({ 
  currentPrice, 
  predictedPrice,
  predictionDate,
  historicalData 
}: PredictionChartProps) {
  const data: ChartDataPoint[] = [];
  
  if (historicalData && historicalData.length > 0) {
    historicalData.forEach((point) => {
      data.push({
        date: new Date(point.date).toLocaleDateString('pt-BR', { 
          month: 'short', 
          day: 'numeric' 
        }),
        historical: point.close,
      });
    });
  }
  
  data.push({ 
    date: 'Hoje', 
    current: currentPrice,
    predicted: currentPrice,
  });
  
  data.push({ 
    date: new Date(predictionDate).toLocaleDateString('pt-BR', { 
      month: 'short', 
      day: 'numeric' 
    }),
    predicted: predictedPrice,
  });

  const allPrices = [
    currentPrice, 
    predictedPrice,
    ...(historicalData?.map(d => d.close) || [])
  ];
  const minPrice = Math.min(...allPrices);
  const maxPrice = Math.max(...allPrices);
  const padding = (maxPrice - minPrice) * 0.2 || 1;

  return (
    <Card className="transition-all duration-300 hover:shadow-lg hover:scale-[1.02]">
      <CardHeader>
        <CardTitle className="text-lg sm:text-xl">
          {historicalData ? 'Histórico e Previsão de Preço' : 'Gráfico de Previsão de Preço'}
        </CardTitle>
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
              tickFormatter={(value) => `R$${value.toFixed(2)}`}
            />
            <Tooltip 
              formatter={(value: number | undefined) => value !== undefined ? [`R$${value.toFixed(2)}`, ''] : ['', '']}
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
              label={{ value: 'Atual', fontSize: 11 }}
            />
            {historicalData && historicalData.length > 0 && (
              <Line
                type="monotone"
                dataKey="historical"
                stroke="#8b5cf6"
                strokeWidth={2}
                name="Histórico "
                dot={false}
                animationDuration={800}
              />
            )}
            <Line
              type="monotone"
              dataKey="current"
              stroke="#3b82f6"
              strokeWidth={3}
              name="Preço Atual"
              dot={{ r: 6 }}
              animationDuration={1000}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#10b981"
              strokeWidth={3}
              strokeDasharray="5 5"
              name="Previsão"
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
