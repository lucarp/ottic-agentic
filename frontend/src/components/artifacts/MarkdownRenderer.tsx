import { marked } from 'marked';
import type { MarkdownArtifactData } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface MarkdownRendererProps {
  data: MarkdownArtifactData;
}

export const MarkdownRenderer = ({ data }: MarkdownRendererProps) => {
  const { content, title } = data;
  const html = marked(content);

  return (
    <Card>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        <div
          className="prose prose-invert max-w-none"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </CardContent>
    </Card>
  );
};
