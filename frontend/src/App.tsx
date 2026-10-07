import React, { useState, useEffect } from "react";
import {
  ChatMessageItem,
  CitationItem,
  ModelsResponse,
} from "./types";
import { fetchModels, askQuestion, streamQuestion } from "./api";
import { Header } from "./components/Header";
import { ChatWindow } from "./components/ChatWindow";
import { ChatInput } from "./components/ChatInput";
import { ArticleBrowser } from "./components/ArticleBrowser";
import { CitationCard } from "./components/CitationCard";
import { Footer } from "./components/Footer";
import { AlertCircle } from "lucide-react";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"chat" | "browser">("chat");
  const [language, setLanguage] = useState<"ar" | "en">("ar");
  const [darkMode, setDarkMode] = useState<boolean>(false);

  // Models state
  const [modelsCatalog, setModelsCatalog] = useState<ModelsResponse | null>(null);
  const [selectedChatModel, setSelectedChatModel] = useState<string>("");
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState<string>("");

  // Chat state
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedCitation, setSelectedCitation] = useState<CitationItem | null>(null);
  const [browserArticleNumber, setBrowserArticleNumber] = useState<number>(1);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  // Sync dark mode class
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  // Sync RTL dir on html tag
  useEffect(() => {
    document.documentElement.dir = language === "ar" ? "rtl" : "ltr";
    document.documentElement.lang = language;
  }, [language]);

  // Fetch models catalog on initial mount
  useEffect(() => {
    fetchModels()
      .then((catalog) => {
        setModelsCatalog(catalog);
        setSelectedChatModel(catalog.defaults.chat_model);
        setSelectedEmbeddingModel(catalog.defaults.embedding_model);
      })
      .catch((err) => {
        console.warn("Could not load /models from backend:", err);
      });
  }, []);

  const handleSendMessage = async (
    question: string,
    options: { topK: number; isStreaming: boolean }
  ) => {
    const userMessageId = `user-${Date.now()}`;
    const userMsg: ChatMessageItem = {
      id: userMessageId,
      role: "user",
      content: question,
      timestamp: new Date().toLocaleTimeString(language === "ar" ? "ar-EG" : "en-US", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    setErrorBanner(null);

    const assistantMessageId = `assistant-${Date.now()}`;

    if (options.isStreaming) {
      // Initialize empty assistant bubble
      const initialAssistantMsg: ChatMessageItem = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        isStreaming: true,
        timestamp: new Date().toLocaleTimeString(language === "ar" ? "ar-EG" : "en-US", {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };
      setMessages((prev) => [...prev, initialAssistantMsg]);

      await streamQuestion(
        {
          question,
          chat_model: selectedChatModel || undefined,
          embedding_model: selectedEmbeddingModel || undefined,
          top_k: options.topK,
          language,
        },
        {
          onToken: (token) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? { ...m, content: m.content + token }
                  : m
              )
            );
          },
          onCitations: (citations) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId ? { ...m, citations } : m
              )
            );
          },
          onDone: (data) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? {
                      ...m,
                      isStreaming: false,
                      latency_ms: (data.latency_ms as number) || undefined,
                      grounding: (data.grounding as any) || undefined,
                    }
                  : m
              )
            );
            setIsLoading(false);
          },
          onError: (err) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMessageId
                  ? {
                      ...m,
                      isStreaming: false,
                      content:
                        m.content ||
                        (language === "ar"
                          ? "عذراً، حدث خطأ أثناء معالجة السؤال."
                          : "Sorry, an error occurred while processing your request."),
                    }
                  : m
              )
            );
            setErrorBanner(err.message);
            setIsLoading(false);
          },
        }
      );
    } else {
      // Non-streaming askQuestion
      try {
        const resp = await askQuestion({
          question,
          chat_model: selectedChatModel || undefined,
          embedding_model: selectedEmbeddingModel || undefined,
          top_k: options.topK,
          language,
        });

        const assistantMsg: ChatMessageItem = {
          id: assistantMessageId,
          role: "assistant",
          content: resp.answer,
          citations: resp.citations,
          grounding: resp.grounding,
          latency_ms: resp.latency_ms,
          timestamp: new Date().toLocaleTimeString(language === "ar" ? "ar-EG" : "en-US", {
            hour: "2-digit",
            minute: "2-digit",
          }),
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err: any) {
        setErrorBanner(err.message || "Query failed");
        const assistantMsg: ChatMessageItem = {
          id: assistantMessageId,
          role: "assistant",
          content:
            language === "ar"
              ? "تعذر الحصول على إجابة، يرجى المحاولة لاحقاً."
              : "Unable to retrieve an answer, please try again later.",
          timestamp: new Date().toLocaleTimeString(language === "ar" ? "ar-EG" : "en-US", {
            hour: "2-digit",
            minute: "2-digit",
          }),
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleOpenArticleInBrowser = (num: number) => {
    setBrowserArticleNumber(num);
    setActiveTab("browser");
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-900 text-slate-850 dark:text-slate-100 font-sans transition-colors duration-200">
      {/* Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        language={language}
        setLanguage={setLanguage}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        chatModels={modelsCatalog?.chat_models || []}
        selectedChatModel={selectedChatModel}
        onSelectChatModel={setSelectedChatModel}
        embeddingModels={modelsCatalog?.embedding_models || []}
        selectedEmbeddingModel={selectedEmbeddingModel}
        onSelectEmbeddingModel={setSelectedEmbeddingModel}
      />

      {/* Error alert toast */}
      {errorBanner && (
        <div className="bg-rose-50 dark:bg-rose-950/60 border-y border-rose-200 dark:border-rose-900 px-4 py-2 text-rose-800 dark:text-rose-300 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-600" />
            <span>{errorBanner}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorBanner(null)}
            className="text-xs hover:underline font-semibold"
          >
            {language === "ar" ? "إغلاق" : "Dismiss"}
          </button>
        </div>
      )}

      {/* Main workspace */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {activeTab === "chat" ? (
          <div className="flex-1 flex flex-col overflow-hidden">
            <ChatWindow
              messages={messages}
              isLoading={isLoading}
              language={language}
              onSelectCitation={setSelectedCitation}
              onOpenArticle={handleOpenArticleInBrowser}
            />
            <ChatInput
              onSend={handleSendMessage}
              isLoading={isLoading}
              language={language}
            />
          </div>
        ) : (
          <ArticleBrowser
            initialArticleNumber={browserArticleNumber}
            language={language}
            onArticleChange={setBrowserArticleNumber}
          />
        )}
      </main>

      {/* Citation Modal */}
      {selectedCitation && (
        <CitationCard
          citation={selectedCitation}
          onClose={() => setSelectedCitation(null)}
          onViewInBrowser={(num: number) => {
            setSelectedCitation(null);
            handleOpenArticleInBrowser(num);
          }}
          language={language}
        />
      )}

      {/* Footer */}
      <Footer language={language} />
    </div>
  );
};
