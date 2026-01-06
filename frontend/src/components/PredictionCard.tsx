import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { PredictionData } from '@/types';

interface PredictionCardProps {
  prediction: PredictionData;
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const isPositive = prediction.change_direction === 'up';
  const isNeutral = prediction.change_direction === 'neutral';
  
  const Icon = isNeutral ? Minus : (isPositive ? TrendingUp : TrendingDown);
  const colorClass = isNeutral 
    ? 'text-gray-500 bg-gray-50' 
    : (isPositive ? 'text-green-600 bg-green-50' : 'text-red-600 bg-red-50');
  
  const confidenceVariant = prediction.confidence === 'high' 
    ? 'default' 
    : prediction.confidence === 'medium' 
    ? 'secondary' 
    : 'destructive';

  return (
    <Card className="transition-all duration-300 hover:shadow-lg">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="text-2xl">{prediction.ticker}</span>
          <Badge variant={confidenceVariant} className="text-sm">
            {prediction.confidence.toUpperCase()} Confidence
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-muted-foreground mb-1">Current Price</p>
              <p className="text-3xl font-bold">${prediction.current_price.toFixed(2)}</p>
            </div>
            
            <div>
              <p className="text-sm text-muted-foreground mb-1">Predicted Price</p>
              <p className="text-3xl font-bold">${prediction.predicted_price.toFixed(2)}</p>
            </div>
          </div>
          
          <div className={`flex items-center justify-center gap-3 p-4 rounded-lg ${colorClass}`}>
            <Icon className="w-8 h-8" />
            <div className="text-center">
              <p className="text-2xl font-bold">
                {prediction.change_percent > 0 ? '+' : ''}
                {prediction.change_percent.toFixed(2)}%
              </p>
              <p className="text-sm font-medium uppercase">
                {prediction.change_direction}
              </p>
            </div>
          </div>
          
          <div className="pt-4 border-t">
            <p className="text-sm text-muted-foreground">Prediction Date</p>
            <p className="text-lg font-semibold">
              {new Date(prediction.prediction_date).toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}
            </p>
          </div>

          <div className="text-xs text-muted-foreground text-center">
            Generated at {new Date(prediction.timestamp).toLocaleString()}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
