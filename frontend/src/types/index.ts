// WebSocket Message Types
export type MessageType =
  | 'user_message'
  | 'assistant_message'
  | 'reasoning'
  | 'reasoning_delta'
  | 'text_delta'
  | 'text_done'
  | 'tool_execution'
  | 'artifact_created'
  | 'error'
  | 'max_turns_exceeded'
  | 'continue_execution';

export interface BaseMessage {
  type: MessageType;
}

export interface UserMessage extends BaseMessage {
  type: 'user_message';
  content: string;
  timestamp?: string;
}

export interface AssistantMessage extends BaseMessage {
  type: 'assistant_message';
  content: string;
  response_id?: string;
}

export interface ReasoningMessage extends BaseMessage {
  type: 'reasoning';
  content: string;
  response_id?: string;
}

export interface ReasoningDeltaMessage extends BaseMessage {
  type: 'reasoning_delta';
  delta: string;
}

export interface TextDeltaMessage extends BaseMessage {
  type: 'text_delta';
  delta: string;
}

export interface TextDoneMessage extends BaseMessage {
  type: 'text_done';
}

export interface ToolExecutionMessage extends BaseMessage {
  type: 'tool_execution';
  tool_name: string;
  status: 'started' | 'completed' | 'failed';
  input?: any;
  output?: any;
}

export interface ArtifactCreatedMessage extends BaseMessage {
  type: 'artifact_created';
  artifact: Artifact;
}

export interface ErrorMessage extends BaseMessage {
  type: 'error';
  error: string;
}

export interface MaxTurnsExceededMessage extends BaseMessage {
  type: 'max_turns_exceeded';
  message: string;
  response_id: string;
}

export interface ContinueExecutionMessage extends BaseMessage {
  type: 'continue_execution';
  content: string;
  previous_response_id: string;
}

export type WebSocketMessage =
  | UserMessage
  | AssistantMessage
  | ReasoningMessage
  | ReasoningDeltaMessage
  | TextDeltaMessage
  | TextDoneMessage
  | ToolExecutionMessage
  | ArtifactCreatedMessage
  | ErrorMessage
  | MaxTurnsExceededMessage
  | ContinueExecutionMessage;

// Artifact Types
export type ArtifactType =
  | 'csv'
  | 'chart'
  | 'code'
  | 'markdown'
  | 'html'
  | 'payment_link'
  | 'fetched_link'
  | 'domain_overview'
  | 'competitor_analysis'
  | 'keyword_research';

export type ArtifactStatus = 'intermediate' | 'final';

export interface Artifact {
  id: string;
  type: ArtifactType;
  status: ArtifactStatus;
  data: any;
  artifact_metadata?: any;
  created_at: string;
}

// CSV Artifact
export interface CsvArtifactData {
  headers: string[];
  rows: any[][];
  title?: string;
  description?: string;
}

// Chart Artifact
export interface ChartArtifactData {
  chart_type: 'bar' | 'line' | 'pie' | 'doughnut' | 'radar';
  labels: string[];
  datasets: ChartDataset[];
  title?: string;
  x_axis_label?: string;
  y_axis_label?: string;
}

export interface ChartDataset {
  label: string;
  data: number[];
  backgroundColor?: string | string[];
  borderColor?: string | string[];
  borderWidth?: number;
}

// Code Artifact
export interface CodeArtifactData {
  code: string;
  language: string;
  title?: string;
  description?: string;
}

// Markdown Artifact
export interface MarkdownArtifactData {
  content: string;
  title?: string;
}

// HTML Artifact
export interface HtmlArtifactData {
  html: string;
  title?: string;
  css?: string;
}

// Payment Link Artifact
export interface PaymentLinkArtifactData {
  amount: number;
  currency: string;
  description: string;
  success_message?: string;
}

// Fetched Link Artifact
export interface FetchedLinkArtifactData {
  url: string;
  title?: string;
  content: string;
  content_type: 'text' | 'markdown';
  fetch_timestamp: string;
  metadata?: {
    author?: string;
    published_date?: string;
  };
  summary?: string;
}

// SEO Domain Overview Artifact
export interface DomainOverviewArtifactData {
  domain: string;
  total_keywords: number;
  organic_traffic: number;
  organic_traffic_value: number;
  paid_keywords: number;
  paid_traffic: number;
  paid_traffic_value: number;
  currency: string;
  title?: string;
}

// SEO Competitor Analysis Artifact
export interface CompetitorAnalysisArtifactData {
  target_domain: string;
  source: string;
  type: 'organic' | 'paid';
  competitors: Competitor[];
  total_competitors: number;
  title?: string;
}

export interface Competitor {
  rank: number;
  domain: string;
  common_keywords: number;
}

// SEO Keyword Research Artifact
export interface KeywordResearchArtifactData {
  analysis_type: 'similar' | 'gap';
  primary_keyword?: string;
  target_domain?: string;
  competitor_domain?: string;
  source: string;
  keywords: KeywordData[];
  total_results: number;
  title?: string;
}

export interface KeywordData {
  keyword: string;
  volume: number;
  cpc: number;
  difficulty: number;
  position?: number;
}
