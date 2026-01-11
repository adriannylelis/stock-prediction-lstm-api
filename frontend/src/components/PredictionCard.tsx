import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { PredictionData } from '@/types';

interface PredictionCardProps {
  prediction: PredictionData;
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const isPositive = prediction.change_direction === 'alta';
  const isNeutral = prediction.change_direction === 'neutra';
  
  const Icon = isNeutral ? Minus : (isPositive ? TrendingUp : TrendingDown);
  const colorClass = isNeutral 
    ? 'text-gray-500 bg-gray-50' 
    : (isPositive ? 'text-green-600 bg-green-50' : 'text-red-600 bg-red-50');
  
  const confidenceVariant = prediction.confidence === 'alta' 
    ? 'default' 
    : prediction.confidence === 'média' 
    ? 'secondary' 
    : 'destructive';

  const confidenceLabel =
    prediction.confidence === 'alta'
      ? 'Confiança Alta'
      : prediction.confidence === 'média'
      ? 'Confiança Média'
      : 'Confiança Baixa';

  return (
    <Card className="transition-all duration-300 hover:shadow-lg hover:scale-[1.02]">
      <CardHeader>
        <CardTitle className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <span className="text-2xl sm:text-3xl">{prediction.ticker}</span>
          <Badge variant={confidenceVariant} className="text-xs sm:text-sm">
            {confidenceLabel}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs sm:text-sm text-muted-foreground mb-1">Preço Atual</p>
              <p className="text-2xl sm:text-3xl font-bold">R${prediction.current_price.toFixed(2)}</p>
            </div>
            
            <div>
              <p className="text-xs sm:text-sm text-muted-foreground mb-1">Preço Previsto</p>
              <p className="text-2xl sm:text-3xl font-bold">R${prediction.predicted_price.toFixed(2)}</p>
            </div>
          </div>
          
          <div className={`flex items-center justify-center gap-3 p-4 rounded-lg transition-colors ${colorClass}`}>
            <Icon className="w-6 h-6 sm:w-8 sm:h-8" />
            <div className="text-center">
              <p className="text-xl sm:text-2xl font-bold">
                {prediction.change_percent > 0 ? '+' : ''}
                {prediction.change_percent.toFixed(2)}%
              </p>
              <p className="text-xs sm:text-sm font-medium uppercase">
                {prediction.change_direction}
              </p>
            </div>
          </div>
          
          <div className="pt-4 border-t">
            <p className="text-xs sm:text-sm text-muted-foreground">Data de Previsão</p>
            <p className="text-base sm:text-lg font-semibold break-words">
              {new Date(prediction.prediction_date).toLocaleDateString('pt-BR', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}
            </p>
          </div>

          <div className="text-xs text-muted-foreground text-center">
            Gerado em {new Date(prediction.timestamp).toLocaleString()}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
