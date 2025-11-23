import { marked } from 'marked';
import type { FetchedLinkArtifactData } from '@/types';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ExternalLink, Calendar, User, Clock } from 'lucide-react';

interface FetchedLinkRendererProps {
  data: FetchedLinkArtifactData;
}

export const FetchedLinkRenderer = ({ data }: FetchedLinkRendererProps) => {
  const { url, title, content, content_type, fetch_timestamp, metadata, summary } = data;

  const renderContent = () => {
    if (content_type === 'markdown') {
      return (
        <div
          className="prose prose-invert max-w-none"
          dangerouslySetInnerHTML={{ __html: marked(content) }}
        />
      );
    } else {
      return (
        <div className="whitespace-pre-wrap">
          {content}
        </div>
      );
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-start justify-between">
            <span>{title || 'Fetched Content'}</span>
          </CardTitle>
          <CardDescription>
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:underline flex items-center gap-2 break-all"
            >
              <ExternalLink className="w-4 h-4 flex-shrink-0" />
              {url}
            </a>
          </CardDescription>

          {/* Metadata */}
          {metadata && (
            <div className="flex flex-wrap gap-4 mt-3 text-sm text-muted-foreground">
              {metadata.author && (
                <div className="flex items-center gap-1">
                  <User className="w-4 h-4" />
                  {metadata.author}
                </div>
              )}
              {metadata.published_date && (
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  {new Date(metadata.published_date).toLocaleDateString()}
                </div>
              )}
              {fetch_timestamp && (
                <div className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  Fetched: {new Date(fetch_timestamp).toLocaleString()}
                </div>
              )}
            </div>
          )}
        </CardHeader>
      </Card>

      {/* Summary */}
      {summary && (
        <Card className="bg-blue-950/30 border-blue-900/50">
          <CardHeader>
            <CardTitle className="text-sm text-blue-400 uppercase tracking-wide">
              Summary
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="leading-relaxed">{summary}</p>
          </CardContent>
        </Card>
      )}

      {/* Content */}
      <Card>
        <CardContent className="pt-6">
          {renderContent()}
        </CardContent>
      </Card>
    </div>
  );
};
