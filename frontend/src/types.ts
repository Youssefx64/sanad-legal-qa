export interface CitationItem {
  article_number: number;
  citation: string;
  text_ar_snippet: string;
  text_en_snippet: string;
  source_page: number;
  is_grounded: boolean;
  is_repealed?: boolean;
}

export interface ChunkRecord {
  chunk_id: string;
  article_number: number;
  book?: string;
  part?: string;
  chapter?: string;
  section?: string;
  topic?: string;
  citation: string;
  text_ar: string;
  text_en: string;
  is_repealed: boolean;
  source_page: number;
}

export interface SearchResult {
  chunk: ChunkRecord;
  score: number;
  source: string;
}

export interface GroundingCheckResult {
  is_grounded: boolean;
  hallucinated_citations?: number[];
  grounded_citations?: number[];
  confidence_score: number;
  reason: string;
}

export interface ChatUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface AskResponse {
  answer: string;
  citations: CitationItem[];
  retrieved_articles: SearchResult[];
  grounding: GroundingCheckResult;
  usage: ChatUsage;
  latency_ms: number;
  chat_model_id: string;
  embedding_model_id: string;
  trace_id: string;
  pii_redacted: boolean;
}

export interface ChatModelInfo {
  id: string;
  label: string;
  provider: string;
  model: string;
  supports_streaming: boolean;
  enabled: boolean;
}

export interface EmbeddingModelInfo {
  id: string;
  label: string;
  provider: string;
  model: string;
  dimension?: number;
  enabled: boolean;
}

export interface ModelsResponse {
  defaults: {
    chat_model: string;
    embedding_model: string;
    judge_model?: string;
  };
  chat_models: ChatModelInfo[];
  embedding_models: EmbeddingModelInfo[];
}

export interface ArticleRecord {
  article_number: number;
  book?: string;
  part?: string;
  chapter?: string;
  section?: string;
  topic?: string;
  text_ar: string;
  text_ar_raw: string;
  text_en: string;
  is_repealed: boolean;
  source_page: number;
  citation: string;
  references: number[];
}

export interface CorpusStatsResponse {
  total_articles: number;
  active_articles: number;
  repealed_articles: number;
  books: string[];
}

export interface ChatMessageItem {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: CitationItem[];
  grounding?: GroundingCheckResult;
  latency_ms?: number;
  isStreaming?: boolean;
  timestamp: string;
}

