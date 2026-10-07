import {
  ArticleRecord,
  AskResponse,
  CitationItem,
  CorpusStatsResponse,
  ModelsResponse,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function fetchModels(): Promise<ModelsResponse> {
  const res = await fetch(`${API_BASE_URL}/models`);
  if (!res.ok) {
    throw new Error(`Failed to fetch models catalog (${res.status})`);
  }
  return res.json();
}

export async function fetchArticle(articleNumber: number): Promise<ArticleRecord> {
  const res = await fetch(`${API_BASE_URL}/articles/${articleNumber}`);
  if (!res.ok) {
    throw new Error(`Article ${articleNumber} not found (${res.status})`);
  }
  return res.json();
}

export async function fetchCorpusStats(): Promise<CorpusStatsResponse> {
  const res = await fetch(`${API_BASE_URL}/corpus/stats`);
  if (!res.ok) {
    throw new Error(`Failed to fetch corpus statistics (${res.status})`);
  }
  return res.json();
}

export async function askQuestion(payload: {
  question: string;
  chat_model?: string;
  embedding_model?: string;
  top_k?: number;
  language?: string;
}): Promise<AskResponse> {
  const res = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Query failed with status ${res.status}`);
  }
  return res.json();
}

export async function streamQuestion(
  payload: {
    question: string;
    chat_model?: string;
    embedding_model?: string;
    top_k?: number;
    language?: string;
  },
  callbacks: {
    onToken: (token: string) => void;
    onCitations: (citations: CitationItem[]) => void;
    onDone: (data: Record<string, unknown>) => void;
    onError: (err: Error) => void;
  }
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Streaming request failed (${response.status})`);
    }

    if (!response.body) {
      throw new Error("No response body available for streaming");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    const processBlock = (block: string) => {
      if (!block.trim()) return;

      let eventType = "message";
      const dataLines: string[] = [];

      const subLines = block.split(/\r?\n/);
      for (const line of subLines) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          const rawData = line.slice(5);
          dataLines.push(rawData.startsWith(" ") ? rawData.slice(1) : rawData);
        }
      }

      const dataStr = dataLines.join("\n");

      if (eventType === "token") {
        callbacks.onToken(dataStr);
      } else if (eventType === "citations") {
        try {
          const parsed = JSON.parse(dataStr);
          callbacks.onCitations(parsed);
        } catch {
          // Ignore parse errors on malformed citations chunk
        }
      } else if (eventType === "done") {
        try {
          const parsed = JSON.parse(dataStr);
          callbacks.onDone(parsed);
        } catch {
          callbacks.onDone({});
        }
      } else if (eventType === "error") {
        try {
          const parsed = JSON.parse(dataStr);
          callbacks.onError(new Error(parsed.detail || "Server stream error"));
        } catch {
          callbacks.onError(new Error(dataStr || "Stream error"));
        }
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      // Keep last incomplete chunk in buffer
      buffer = blocks.pop() || "";

      for (const block of blocks) {
        processBlock(block);
      }
    }

    // Flush any remaining complete blocks in buffer
    buffer += decoder.decode();
    if (buffer.trim()) {
      const blocks = buffer.split(/\r?\n\r?\n/);
      for (const block of blocks) {
        processBlock(block);
      }
    }
  } catch (err) {
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
  }
}

