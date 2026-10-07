import React, { useState, useRef, useEffect } from "react";
import { Send, SlidersHorizontal, Sparkles } from "lucide-react";

interface ChatInputProps {
  onSend: (message: string, options: { topK: number; isStreaming: boolean }) => void;
  isLoading: boolean;
  language: "ar" | "en";
}

const SAMPLE_QUESTIONS_AR = [
  "ما هي شروط صحة الرضا في العقد؟",
  "ما هو أثر القوة القاهرة على تنفيذ الالتزام؟",
  "ما هي أحكام فسخ العقد الملزم للجانبين وفقاً للمادة 157؟",
  "ما هي الحقوق التي تسقط بالتقادم المسقط ومددها؟",
];

const SAMPLE_QUESTIONS_EN = [
  "What constitutes force majeure under the Egyptian Civil Code?",
  "What are the consequences of breach of contract under Article 157?",
  "What are the rules regarding contract nullity and consent?",
  "What is the statutory period of limitation (prescription) for claims?",
];

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  isLoading,
  language,
}) => {
  const [text, setText] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [topK, setTopK] = useState(5);
  const [isStreaming, setIsStreaming] = useState(true);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        180
      )}px`;
    }
  }, [text]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!text.trim() || isLoading) return;
    onSend(text.trim(), { topK, isStreaming });
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const sampleQuestions = language === "ar" ? SAMPLE_QUESTIONS_AR : SAMPLE_QUESTIONS_EN;

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-4">
      {/* Sample Question Pills */}
      <div className="mb-3 flex items-center gap-2 overflow-x-auto pb-1 no-scrollbar text-xs">
        <span className="text-slate-400 dark:text-slate-500 font-medium flex items-center gap-1 shrink-0">
          <Sparkles className="w-3.5 h-3.5 text-amber-500" />
          {language === "ar" ? "أمثلة شائعة:" : "Sample queries:"}
        </span>
        {sampleQuestions.map((q, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => setText(q)}
            className="shrink-0 px-3 py-1.5 rounded-full bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 transition-colors border border-slate-200/80 dark:border-slate-700/80"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Main input card */}
      <div className="relative rounded-2xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-lg shadow-slate-100/50 dark:shadow-none focus-within:ring-2 focus-within:ring-emerald-500/30 focus-within:border-emerald-500 transition-all">
        <form onSubmit={handleSubmit} className="flex flex-col">
          <div className="flex items-end px-3 py-2.5 gap-2">
            <textarea
              ref={textareaRef}
              rows={1}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                language === "ar"
                  ? "اطرح سؤالك القانوني استناداً إلى القانون المدني المصري..."
                  : "Ask a legal question based on the Egyptian Civil Code..."
              }
              className="w-full resize-none bg-transparent outline-none py-1.5 px-2 text-sm text-slate-800 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 max-h-44 leading-relaxed"
              dir="auto"
              disabled={isLoading}
            />

            <div className="flex items-center gap-1.5 shrink-0 mb-0.5">
              <button
                type="button"
                onClick={() => setShowSettings(!showSettings)}
                title={language === "ar" ? "خيارات البحث" : "Search Options"}
                className={`p-2 rounded-xl transition-colors ${
                  showSettings
                    ? "bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400"
                    : "text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                }`}
              >
                <SlidersHorizontal className="w-4 h-4" />
              </button>

              <button
                type="submit"
                disabled={!text.trim() || isLoading}
                className="p-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-200 dark:disabled:bg-slate-700 text-white disabled:text-slate-400 dark:disabled:text-slate-500 transition-all shadow-sm flex items-center justify-center"
                title={language === "ar" ? "إرسال" : "Send"}
              >
                <Send className={`w-4 h-4 ${language === "ar" ? "rotate-180" : ""}`} />
              </button>
            </div>
          </div>

          {/* Quick Settings Drawer */}
          {showSettings && (
            <div className="border-t border-slate-100 dark:border-slate-750 px-4 py-3 bg-slate-50/70 dark:bg-slate-800/70 rounded-b-2xl flex flex-wrap items-center justify-between gap-4 text-xs">
              <div className="flex items-center gap-3">
                <label className="text-slate-600 dark:text-slate-400 font-medium">
                  {language === "ar" ? "عدد المواد المسترجعة (Top-K):" : "Retrieved Articles (Top-K):"}
                </label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={1}
                    max={15}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-24 accent-emerald-600 cursor-pointer"
                  />
                  <span className="font-mono font-semibold text-emerald-700 dark:text-emerald-400 w-4 text-center">
                    {topK}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-slate-600 dark:text-slate-400 font-medium cursor-pointer flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={isStreaming}
                    onChange={(e) => setIsStreaming(e.target.checked)}
                    className="rounded text-emerald-600 focus:ring-emerald-500 h-3.5 w-3.5 cursor-pointer accent-emerald-600"
                  />
                  <span>
                    {language === "ar" ? "تدفق الإجابة لحظياً (Streaming)" : "Stream response tokens"}
                  </span>
                </label>
              </div>
            </div>
          )}
        </form>
      </div>
    </div>
  );
};
