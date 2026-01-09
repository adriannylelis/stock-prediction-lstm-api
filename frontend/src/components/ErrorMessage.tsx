import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle } from 'lucide-react';

interface ErrorMessageProps {
  title?: string;
  message: string;
}

export function ErrorMessage({ 
  title = 'Error', 
  message 
}: ErrorMessageProps) {
  return (
    <Alert variant="destructive" className="animate-in fade-in-50">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription className="text-sm">
        {message}
      </AlertDescription>
    </Alert>
  );
}
