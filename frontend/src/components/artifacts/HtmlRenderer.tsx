import type { HtmlArtifactData } from '@/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface HtmlRendererProps {
  data: HtmlArtifactData;
}

export const HtmlRenderer = ({ data }: HtmlRendererProps) => {
  const { html, title, css } = data;

  return (
    <Card>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        {css && <style>{css}</style>}
        <div
          className="bg-white text-black p-4 rounded-md"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </CardContent>
    </Card>
  );
};
