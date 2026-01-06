import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { Stock } from '@/types';

interface StockSelectorProps {
  stocks: Stock[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export function StockSelector({ 
  stocks, 
  value, 
  onChange, 
  disabled = false 
}: StockSelectorProps) {
  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger className="w-full transition-all duration-200 hover:border-primary">
        <SelectValue placeholder="Select a stock ticker" />
      </SelectTrigger>
      <SelectContent>
        {stocks.map((stock) => (
          <SelectItem 
            key={stock.symbol} 
            value={stock.symbol}
            className="cursor-pointer transition-colors"
          >
            <div className="flex items-center gap-2">
              <span className="font-semibold">{stock.symbol}</span>
              <span className="text-muted-foreground text-xs md:text-sm">- {stock.name}</span>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
