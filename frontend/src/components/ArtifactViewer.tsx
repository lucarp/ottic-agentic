import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import type { Artifact } from '@/types';
import { CsvRenderer } from './artifacts/CsvRenderer';
import { ChartRenderer } from './artifacts/ChartRenderer';
import { CodeRenderer } from './artifacts/CodeRenderer';
import { MarkdownRenderer } from './artifacts/MarkdownRenderer';
import { HtmlRenderer } from './artifacts/HtmlRenderer';
import { PaymentLinkRenderer } from './artifacts/PaymentLinkRenderer';
import { FetchedLinkRenderer } from './artifacts/FetchedLinkRenderer';
import { DomainOverviewRenderer } from './artifacts/DomainOverviewRenderer';
import { CompetitorAnalysisRenderer } from './artifacts/CompetitorAnalysisRenderer';
import { KeywordResearchRenderer } from './artifacts/KeywordResearchRenderer';
import { Package } from 'lucide-react';

interface ArtifactViewerProps {
  artifacts: Artifact[];
}

export const ArtifactViewer = ({ artifacts }: ArtifactViewerProps) => {
  const [activeTab, setActiveTab] = useState<string>('');

  // Set active tab to latest artifact when new one arrives
  if (artifacts.length > 0 && !activeTab) {
    setActiveTab(artifacts[artifacts.length - 1].id);
  }

  const renderArtifact = (artifact: Artifact) => {
    switch (artifact.type) {
      case 'csv':
        return <CsvRenderer data={artifact.data} />;
      case 'chart':
        return <ChartRenderer data={artifact.data} />;
      case 'code':
        return <CodeRenderer data={artifact.data} />;
      case 'markdown':
        return <MarkdownRenderer data={artifact.data} />;
      case 'html':
        return <HtmlRenderer data={artifact.data} />;
      case 'payment_link':
        return <PaymentLinkRenderer data={artifact.data} />;
      case 'fetched_link':
        return <FetchedLinkRenderer data={artifact.data} />;
      case 'domain_overview':
        return <DomainOverviewRenderer data={artifact.data} />;
      case 'competitor_analysis':
        return <CompetitorAnalysisRenderer data={artifact.data} />;
      case 'keyword_research':
        return <KeywordResearchRenderer data={artifact.data} />;
      default:
        return (
          <pre className="bg-muted p-4 rounded-md overflow-auto">
            {JSON.stringify(artifact.data, null, 2)}
          </pre>
        );
    }
  };

  if (artifacts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <Package className="w-16 h-16 mb-4" />
        <h3 className="text-xl font-semibold mb-2">No artifacts yet</h3>
        <p className="text-sm">Ask the agent to create an artifact to see it here</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-border p-4">
        <h2 className="text-lg font-semibold">Artifacts</h2>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <div className="border-b border-border overflow-x-auto">
          <TabsList className="h-auto p-2 bg-transparent w-full justify-start rounded-none">
            {artifacts.map((artifact) => (
              <TabsTrigger
                key={artifact.id}
                value={artifact.id}
                className="flex items-center gap-2 data-[state=active]:bg-muted"
              >
                <span className="uppercase text-xs font-semibold">{artifact.type}</span>
                <Badge
                  variant={artifact.status === 'final' ? 'default' : 'secondary'}
                  className="text-[10px]"
                >
                  {artifact.status}
                </Badge>
                <span className="text-[10px] text-muted-foreground">
                  {artifact.id.substring(0, 8)}
                </span>
              </TabsTrigger>
            ))}
          </TabsList>
        </div>

        <ScrollArea className="flex-1">
          {artifacts.map((artifact) => (
            <TabsContent key={artifact.id} value={artifact.id} className="p-6 m-0">
              {renderArtifact(artifact)}
            </TabsContent>
          ))}
        </ScrollArea>
      </Tabs>
    </div>
  );
};
