import type { PaymentLinkArtifactData } from '@/types';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface PaymentLinkRendererProps {
  data: PaymentLinkArtifactData;
}

export const PaymentLinkRenderer = ({ data }: PaymentLinkRendererProps) => {
  const { amount, currency, description, success_message } = data;

  return (
    <Card className="max-w-md mx-auto">
      <CardHeader>
        <CardTitle>Payment Request</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="text-center space-y-4">
        <div className="text-5xl font-bold text-green-500">
          ${amount.toFixed(2)}
        </div>
        <div className="text-sm text-muted-foreground uppercase">
          {currency}
        </div>
        <Button
          className="w-full bg-blue-600 hover:bg-blue-700"
          onClick={() => alert('Payment integration not implemented in POC')}
        >
          Pay with Stripe
        </Button>
        {success_message && (
          <p className="text-xs text-muted-foreground mt-4">
            {success_message}
          </p>
        )}
      </CardContent>
    </Card>
  );
};
