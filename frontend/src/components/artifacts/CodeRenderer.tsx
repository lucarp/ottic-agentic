import { useEffect, useRef } from 'react';
import hljs from 'highlight.js';
import 'highlight.js/styles/atom-one-dark.css';
import type { CodeArtifactData } from '@/types';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface CodeRendererProps {
  data: CodeArtifactData;
}

export const CodeRenderer = ({ data }: CodeRendererProps) => {
  const { code, language, title, description } = data;
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (codeRef.current) {
      hljs.highlightElement(codeRef.current);
    }
  }, [code, language]);

  return (
    <Card>
      {(title || description) && (
        <CardHeader>
          {title && <CardTitle>{title}</CardTitle>}
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
      )}
      <CardContent>
        <pre className="rounded-md border border-border overflow-x-auto">
          <code ref={codeRef} className={`language-${language}`}>
            {code}
          </code>
        </pre>
      </CardContent>
    </Card>
  );
};
