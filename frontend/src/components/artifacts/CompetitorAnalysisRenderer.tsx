import type { CompetitorAnalysisArtifactData } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Target, Crown } from 'lucide-react';

interface CompetitorAnalysisRendererProps {
  data: CompetitorAnalysisArtifactData;
}

export const CompetitorAnalysisRenderer = ({ data }: CompetitorAnalysisRendererProps) => {
  const { target_domain, source, type, competitors, total_competitors, title } = data;

  return (
    <div className="space-y-4">
      {title && (
        <h2 className="text-2xl font-bold border-b-2 border-purple-500 pb-3">{title}</h2>
      )}

      {/* Summary Header */}
      <Card className="bg-gradient-to-br from-purple-950/50 to-background border-purple-900/50">
        <CardHeader>
          <div className="flex items-center gap-3">
            <Target className="w-6 h-6 text-purple-400" />
            <CardTitle>
              Found {total_competitors} {type} competitors
            </CardTitle>
          </div>
          <div className="text-sm text-muted-foreground mt-2">
            Competing with{' '}
            <span className="text-purple-400 font-semibold">{target_domain}</span> in{' '}
            <span className="text-purple-400 font-semibold">{source.toUpperCase()}</span>
          </div>
        </CardHeader>
      </Card>

      {/* Competitors Table */}
      {competitors.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            No competitors found
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-4 py-3 text-left text-sm font-semibold text-purple-400">
                      Rank
                    </th>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-purple-400">
                      Competitor Domain
                    </th>
                    <th className="px-4 py-3 text-right text-sm font-semibold text-purple-400">
                      Common Keywords
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {competitors.map((comp, index) => {
                    const isTopCompetitor = comp.rank === 1;
                    return (
                      <tr
                        key={index}
                        className={`border-b border-border/50 ${
                          isTopCompetitor ? 'bg-purple-950/30' : ''
                        }`}
                      >
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            {isTopCompetitor && <Crown className="w-4 h-4 text-purple-400" />}
                            <span
                              className={`font-semibold ${
                                isTopCompetitor ? 'text-purple-400' : ''
                              }`}
                            >
                              #{comp.rank}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <a
                            href={`https://${comp.domain}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className={`hover:underline ${
                              isTopCompetitor ? 'text-purple-300' : 'text-blue-400'
                            }`}
                          >
                            {comp.domain}
                          </a>
                        </td>
                        <td className="px-4 py-3 text-right font-semibold">
                          {comp.common_keywords.toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
