import type { KeywordResearchArtifactData } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Search } from 'lucide-react';

interface KeywordResearchRendererProps {
  data: KeywordResearchArtifactData;
}

export const KeywordResearchRenderer = ({ data }: KeywordResearchRendererProps) => {
  const {
    analysis_type,
    primary_keyword,
    target_domain,
    competitor_domain,
    source,
    keywords,
    total_results,
    title,
  } = data;

  const getDifficultyColor = (difficulty: number) => {
    if (difficulty <= 30) return 'bg-green-500/20 text-green-400 border-green-500';
    if (difficulty <= 60) return 'bg-yellow-500/20 text-yellow-400 border-yellow-500';
    return 'bg-red-500/20 text-red-400 border-red-500';
  };

  const getDifficultyLabel = (difficulty: number) => {
    if (difficulty <= 30) return 'Easy';
    if (difficulty <= 60) return 'Medium';
    return 'Hard';
  };

  return (
    <div className="space-y-4">
      {title && <h2 className="text-2xl font-bold border-b-2 border-green-500 pb-3">{title}</h2>}

      {/* Summary Header */}
      <Card className="bg-gradient-to-br from-green-950/50 to-background border-green-900/50">
        <CardHeader>
          <div className="flex items-center gap-3">
            <Search className="w-6 h-6 text-green-400" />
            <CardTitle>Found {total_results} keywords</CardTitle>
          </div>
          <div className="text-sm text-muted-foreground mt-2">
            {analysis_type === 'similar' && primary_keyword && (
              <>
                Similar to:{' '}
                <span className="text-green-400 font-semibold">{primary_keyword}</span>
              </>
            )}
            {analysis_type === 'gap' && competitor_domain && target_domain && (
              <>
                Keywords where{' '}
                <span className="text-red-400 font-semibold">{competitor_domain}</span> ranks but{' '}
                <span className="text-blue-400 font-semibold">{target_domain}</span> doesn't
              </>
            )}
            {' • '}
            Region: <span className="text-green-400 font-semibold">{source.toUpperCase()}</span>
          </div>
        </CardHeader>
      </Card>

      {/* Keywords Table */}
      {keywords.length === 0 ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            No keywords found
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-4 py-3 text-left text-sm font-semibold text-green-400">
                      Keyword
                    </th>
                    <th className="px-4 py-3 text-right text-sm font-semibold text-green-400">
                      Volume
                    </th>
                    <th className="px-4 py-3 text-right text-sm font-semibold text-green-400">
                      CPC
                    </th>
                    <th className="px-4 py-3 text-center text-sm font-semibold text-green-400">
                      Difficulty
                    </th>
                    {analysis_type === 'gap' && (
                      <th className="px-4 py-3 text-center text-sm font-semibold text-green-400">
                        Position
                      </th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {keywords.map((kw, index) => (
                    <tr
                      key={index}
                      className="border-b border-border/50 hover:bg-muted/20"
                    >
                      <td className="px-4 py-3 font-medium">{kw.keyword}</td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        {kw.volume.toLocaleString()}
                      </td>
                      <td className="px-4 py-3 text-right text-muted-foreground">
                        ${kw.cpc.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <Badge
                          className={`${getDifficultyColor(kw.difficulty)} border`}
                          variant="outline"
                        >
                          {kw.difficulty} - {getDifficultyLabel(kw.difficulty)}
                        </Badge>
                      </td>
                      {analysis_type === 'gap' && (
                        <td className="px-4 py-3 text-center font-semibold">
                          {kw.position ? `#${kw.position}` : '-'}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Legend Footer */}
      <Card className="bg-muted/30">
        <CardContent className="py-3">
          <div className="text-xs text-muted-foreground">
            <strong className="text-foreground">Legend:</strong>{' '}
            <span className="text-green-400">Easy (0-30)</span> •{' '}
            <span className="text-yellow-400">Medium (31-60)</span> •{' '}
            <span className="text-red-400">Hard (61-100)</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
