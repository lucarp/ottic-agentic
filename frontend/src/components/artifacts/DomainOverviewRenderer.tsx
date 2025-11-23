import type { DomainOverviewArtifactData } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { TrendingUp, DollarSign, Target } from 'lucide-react';

interface DomainOverviewRendererProps {
  data: DomainOverviewArtifactData;
}

export const DomainOverviewRenderer = ({ data }: DomainOverviewRendererProps) => {
  const {
    domain,
    total_keywords,
    organic_traffic,
    organic_traffic_value,
    paid_keywords,
    paid_traffic,
    paid_traffic_value,
    currency,
    title,
  } = data;

  const formatNumber = (num: number) => num.toLocaleString();
  const formatCurrency = (num: number) =>
    `${currency} ${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  const totalTraffic = organic_traffic + paid_traffic;
  const totalValue = organic_traffic_value + paid_traffic_value;

  return (
    <div className="space-y-4">
      {title && <h2 className="text-2xl font-bold border-b-2 border-primary pb-3">{title}</h2>}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Organic Traffic Card */}
        <Card className="bg-gradient-to-br from-green-950/50 to-background border-green-900/50">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-3">
              <div className="bg-green-500 p-2 rounded-lg">
                <TrendingUp className="w-5 h-5 text-white" />
              </div>
              <CardTitle className="text-sm text-green-400 uppercase tracking-wide">
                Organic Traffic
              </CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-2">{formatNumber(organic_traffic)}</div>
            <div className="text-sm text-muted-foreground mb-1">
              {formatNumber(total_keywords)} keywords
            </div>
            <div className="text-lg font-semibold text-green-400">
              {formatCurrency(organic_traffic_value)}/mo
            </div>
          </CardContent>
        </Card>

        {/* Paid Traffic Card */}
        <Card className="bg-gradient-to-br from-yellow-950/50 to-background border-yellow-900/50">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-3">
              <div className="bg-yellow-500 p-2 rounded-lg">
                <DollarSign className="w-5 h-5 text-white" />
              </div>
              <CardTitle className="text-sm text-yellow-400 uppercase tracking-wide">
                Paid Traffic
              </CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-2">{formatNumber(paid_traffic)}</div>
            <div className="text-sm text-muted-foreground mb-1">
              {formatNumber(paid_keywords)} keywords
            </div>
            <div className="text-lg font-semibold text-yellow-400">
              {formatCurrency(paid_traffic_value)}/mo
            </div>
          </CardContent>
        </Card>

        {/* Total Card */}
        <Card className="bg-gradient-to-br from-blue-950/50 to-background border-blue-900/50">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-3">
              <div className="bg-blue-500 p-2 rounded-lg">
                <Target className="w-5 h-5 text-white" />
              </div>
              <CardTitle className="text-sm text-blue-400 uppercase tracking-wide">
                Total Estimated
              </CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold mb-2">{formatNumber(totalTraffic)}</div>
            <div className="text-sm text-muted-foreground mb-1">visitors/month</div>
            <div className="text-lg font-semibold text-blue-400">
              {formatCurrency(totalValue)}/mo
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Domain Info Footer */}
      <Card className="bg-muted/30 border-primary/30">
        <CardContent className="py-3">
          <div className="text-sm text-muted-foreground">
            Analyzing domain: <span className="text-primary font-semibold">{domain}</span>
          </div>
          <div className="text-xs text-muted-foreground mt-1">Data source: SE Ranking API</div>
        </CardContent>
      </Card>
    </div>
  );
};
