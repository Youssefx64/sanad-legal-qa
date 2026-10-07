import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { App } from "../App";

// Mock fetch for /models and /articles/1 and /corpus/stats
const mockModelsResponse = {
  defaults: {
    chat_model: "openrouter-qwen",
    embedding_model: "openrouter-text-embedding-3-small",
  },
  chat_models: [
    {
      id: "openrouter-qwen",
      label: "Qwen 2.5 72B Instruct",
      provider: "openrouter",
      model: "qwen/qwen-2.5-72b-instruct",
      supports_streaming: true,
      enabled: true,
    },
    {
      id: "ollama-qwen",
      label: "Ollama Qwen 2.5 7B",
      provider: "ollama",
      model: "qwen2.5:7b-instruct",
      supports_streaming: true,
      enabled: false,
    },
  ],
  embedding_models: [
    {
      id: "openrouter-text-embedding-3-small",
      label: "OpenRouter OpenAI Text Embedding 3 Small",
      provider: "openrouter",
      model: "openai/text-embedding-3-small",
      dimension: 1536,
      enabled: true,
    },
  ],
};

const mockArticleResponse = {
  article_number: 1,
  book: "General Provisions",
  part: "Chapter 1",
  chapter: "Section 1",
  section: "Sources of Law",
  topic: "Sources of Law",
  text_ar: "تسري النصوص التشريعية على جميع المسائل التي تتناولها هذه النصوص في لفظها أو في فحواها.",
  text_ar_raw: "تسري النصوص التشريعية...",
  text_en: "Legislative provisions govern all matters to which they relate in letter or spirit.",
  is_repealed: false,
  source_page: 1,
  citation: "Egyptian Civil Code, Article 1",
  references: [],
};

const mockCorpusStats = {
  total_articles: 1149,
  active_articles: 1100,
  repealed_articles: 49,
  books: ["General Provisions", "Obligations or Personal Rights"],
};

describe("Sanad Frontend Application", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        if (url.includes("/models")) {
          return Promise.resolve({
            ok: true,
            json: async () => mockModelsResponse,
          });
        }
        if (url.includes("/articles/1")) {
          return Promise.resolve({
            ok: true,
            json: async () => mockArticleResponse,
          });
        }
        if (url.includes("/corpus/stats")) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCorpusStats,
          });
        }
        return Promise.reject(new Error(`Unhandled URL: ${url}`));
      })
    );
  });

  it("renders header title and welcome screen", async () => {
    render(<App />);

    expect(screen.getByText("سَنَد")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/مرحباً بك في سند/i)).toBeInTheDocument();
    });
  });

  it("switches language between Arabic and English", async () => {
    render(<App />);

    const langButton = screen.getByTitle("Switch to English");
    fireEvent.click(langButton);

    await waitFor(() => {
      expect(screen.getByText(/Welcome to Sanad Legal Q&A/i)).toBeInTheDocument();
    });
  });

  it("switches between Chat and Article Browser tabs", async () => {
    render(<App />);

    const browserTab = screen.getByText("تصفح نصوص القانون");
    fireEvent.click(browserTab);

    await waitFor(() => {
      expect(screen.getByText("الانتقال إلى مادة")).toBeInTheDocument();
      expect(screen.getByText("إحصاءات المدونة")).toBeInTheDocument();
    });
  });
});
