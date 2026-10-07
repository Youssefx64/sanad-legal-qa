import React from "react";
import {
  ChatMessageItem,
  CitationItem,
} from "../types";
import {
  Scale,
  ShieldCheck,
  ShieldAlert,
  Clock,
  Sparkles,
  BookOpen,
  User,
  ExternalLink,
  Search,
  Loader2,
} from "lucide-react";

interface ChatWindowProps {
  messages: ChatMessageItem[];
  isLoading: boolean;
  language: "ar" | "en";
  onSelectCitation: (citation: CitationItem) => void;
  onOpenArticle: (articleNumber: number) => void;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading,
  language,
  onSelectCitation,
  onOpenArticle,
}) => {
  // Helper to parse text and turn [المادة N] or [Article N] into clickable chips
  const renderFormattedText = (
    text: string,
    citations?: CitationItem[]
  ) => {
    // Regex for [المادة N] or [Article N]
    const regex = /\[(?:المادة|Article)\s+(\d+)\]/gi;
    const parts: (string | JSX.Element)[] = [];
    let lastIdx = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(text)) !== null) {
      const matchStart = match.index;
      const matchEnd = regex.lastIndex;
      const articleNum = parseInt(match[1], 10);

      if (matchStart > lastIdx) {
        parts.push(text.substring(lastIdx, matchStart));
      }

      const matchingCitation = citations?.find(
        (c) => c.article_number === articleNum
      );

      parts.push(
        <button
          key={`chip-${matchStart}`}
          type="button"
          onClick={() => {
            if (matchingCitation) {
              onSelectCitation(matchingCitation);
            } else {
              onOpenArticle(articleNum);
            }
          }}
          className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md bg-emerald-100 hover:bg-emerald-200 dark:bg-emerald-950/60 dark:hover:bg-emerald-900/80 text-emerald-800 dark:text-emerald-300 font-semibold text-xs transition-colors border border-emerald-300 dark:border-emerald-800"
          title={language === "ar" ? `عرض المادة ${articleNum}` : `View Article ${articleNum}`}
        >
          <BookOpen className="w-3 h-3" />
          <span>{match[0]}</span>
        </button>
      );

      lastIdx = matchEnd;
    }

    if (lastIdx < text.length) {
      parts.push(text.substring(lastIdx));
    }

    return (
      <div className="whitespace-pre-wrap leading-relaxed text-sm">
        {parts}
      </div>
    );
  };

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-2xl mx-auto">
        <div className="w-16 h-16 rounded-2xl bg-emerald-100 dark:bg-emerald-950/50 flex items-center justify-center mb-5 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800 shadow-sm">
          <Scale className="w-8 h-8" />
        </div>
        <h2 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2">
          {language === "ar"
            ? "مرحباً بك في سند (Sanad)"
            : "Welcome to Sanad Legal Q&A"}
        </h2>
        <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed mb-6 max-w-lg">
          {language === "ar"
            ? "المساعد القانوني الذكي للقانون المدني المصري (قانون رقم 131 لسنة 1948). يجيب سند بدقة مع توثيق أرقام المواد ومطابقة النصوص دون تخمين."
            : "Arabic legal Q&A over the Egyptian Civil Code (Law No. 131 of 1948). Sanad cites exact article numbers and strictly refrains from hallucination."}
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full text-left text-xs">
          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/60 dark:bg-slate-850/60 shadow-sm">
            <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-semibold mb-1">
              <ShieldCheck className="w-4 h-4" />
              <span>{language === "ar" ? "توثيق قطعي" : "Strict Grounding"}</span>
            </div>
            <p className="text-slate-500 dark:text-slate-400">
              {language === "ar"
                ? "إجابات مبنية حصراً على نصوص المواد مع كشف الاستشهادات الوهمية."
                : "Answers cite retrieved articles only; refusal triggered when ungrounded."}
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/60 dark:bg-slate-850/60 shadow-sm">
            <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-semibold mb-1">
              <Sparkles className="w-4 h-4" />
              <span>{language === "ar" ? "1,149 مادة كاملة" : "1,149 Articles"}</span>
            </div>
            <p className="text-slate-500 dark:text-slate-400">
              {language === "ar"
                ? "فهرسة ثنائية اللغة لجميع مواد القانون، مع تصنيف المواد الملغاة تلقائياً."
                : "Bilingual index across all books with tracked repealed articles."}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6 max-w-4xl mx-auto w-full">
      {messages.map((msg) => {
        const isUser = msg.role === "user";

        return (
          <div
            key={msg.id}
            className={`flex gap-3 ${
              isUser ? "flex-row-reverse" : "flex-row"
            } items-start`}
          >
            {/* Avatar */}
            <div
              className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 text-sm shadow-sm ${
                isUser
                  ? "bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200"
                  : "bg-emerald-600 text-white shadow-emerald-600/20"
              }`}
            >
              {isUser ? <User className="w-4 h-4" /> : <Scale className="w-4 h-4" />}
            </div>

            {/* Bubble Container */}
            <div
              className={`flex flex-col max-w-[85%] sm:max-w-[78%] ${
                isUser ? "items-end" : "items-start"
              }`}
            >
              {/* Message Bubble */}
              <div
                className={`rounded-2xl px-4 py-3.5 shadow-sm text-sm ${
                  isUser
                    ? "bg-emerald-600 text-white rounded-tr-sm"
                    : "bg-white dark:bg-slate-800 text-slate-850 dark:text-slate-100 border border-slate-200 dark:border-slate-700 rounded-tl-sm"
                }`}
              >
                {/* Meta header for assistant */}
                {!isUser && (
                  <div className="flex flex-wrap items-center gap-2 mb-2 pb-2 border-b border-slate-100 dark:border-slate-700 text-xs">
                    {/* Grounding badge */}
                    {msg.grounding && (
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full font-medium ${
                          msg.grounding.is_grounded
                            ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800"
                            : "bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-400 border border-amber-200 dark:border-amber-800"
                        }`}
                      >
                        {msg.grounding.is_grounded ? (
                          <>
                            <ShieldCheck className="w-3 h-3" />
                            <span>{language === "ar" ? "موثق قانونياً" : "Grounded"}</span>
                          </>
                        ) : (
                          <>
                            <ShieldAlert className="w-3 h-3" />
                            <span>{language === "ar" ? "تحذير تأصيل" : "Ungrounded"}</span>
                          </>
                        )}
                      </span>
                    )}

                    {/* Latency */}
                    {msg.latency_ms && (
                      <span className="inline-flex items-center gap-1 text-slate-400 dark:text-slate-500 font-mono text-[11px]">
                        <Clock className="w-3 h-3" />
                        <span>{msg.latency_ms} ms</span>
                      </span>
                    )}
                  </div>
                )}

                {/* Content */}
                {isUser ? (
                  <div className="whitespace-pre-wrap leading-relaxed">
                    {msg.content}
                  </div>
                ) : (
                  <div>
                    {msg.isStreaming && !msg.content ? (
                      <div className="py-2.5 px-3 rounded-2xl bg-gradient-to-r from-emerald-50/70 via-teal-50/40 to-slate-50/60 dark:from-emerald-950/40 dark:via-teal-950/20 dark:to-slate-900/40 border border-emerald-200/80 dark:border-emerald-800/60 shadow-sm space-y-3">
                        <div className="flex items-center gap-2.5">
                          <div className="relative flex items-center justify-center w-8 h-8 rounded-xl bg-emerald-600 text-white shadow-md shadow-emerald-600/30 shrink-0">
                            <Scale className="w-4 h-4 animate-spin-slow" />
                            <span className="absolute -top-0.5 -right-0.5 flex h-2.5 w-2.5">
                              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
                            </span>
                          </div>
                          <div>
                            <div className="text-xs font-bold text-emerald-900 dark:text-emerald-200 flex items-center gap-1.5">
                              <span>
                                {language === "ar"
                                  ? "سند يُفكّر ويُحلل السند القانوني..."
                                  : "Sanad is analyzing legal provisions..."}
                              </span>
                              <Sparkles className="w-3.5 h-3.5 text-amber-500 animate-pulse" />
                            </div>
                            <p className="text-[11px] text-slate-500 dark:text-slate-400">
                              {language === "ar"
                                ? "البحث في 1,149 مادة ومطابقة النصوص والتأصيل"
                                : "Searching 1,149 articles & matching statutory texts"}
                            </p>
                          </div>
                        </div>

                        {/* Animated Step-by-Step Legal Retrieval Indicator */}
                        <div className="p-2.5 rounded-xl bg-white/70 dark:bg-slate-900/60 border border-emerald-100 dark:border-emerald-900/40 space-y-2">
                          <div className="flex items-center justify-between text-[11px] text-slate-600 dark:text-slate-300">
                            <div className="flex items-center gap-2">
                              <Search className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
                              <span>
                                {language === "ar"
                                  ? "استرجاع المواد ذات الصلة والتحقق من سريانها..."
                                  : "Retrieving relevant statutes & checking validity..."}
                              </span>
                            </div>
                            <Loader2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 animate-spin" />
                          </div>
                          <div className="w-full bg-slate-200 dark:bg-slate-700 h-1.5 rounded-full overflow-hidden">
                            <div className="h-full bg-emerald-500 rounded-full animate-shimmer w-full" />
                          </div>
                        </div>
                      </div>
                    ) : (
                      <>
                        {renderFormattedText(msg.content, msg.citations)}
                        {msg.isStreaming && (
                          <span className="inline-block w-2 h-4 ml-1 bg-emerald-500 rounded-sm animate-pulse align-middle shadow-sm shadow-emerald-500/50" />
                        )}
                      </>
                    )}
                  </div>
                )}

                {/* Structured Citations Cards */}
                {!isUser && msg.citations && msg.citations.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-700">
                    <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 mb-2 flex items-center gap-1">
                      <BookOpen className="w-3.5 h-3.5" />
                      <span>{language === "ar" ? "المستندات المستشهد بها:" : "Cited Articles:"}</span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {msg.citations.map((cite) => (
                        <div
                          key={cite.article_number}
                          onClick={() => onSelectCitation(cite)}
                          className="p-2.5 rounded-xl border border-slate-200 dark:border-slate-700 hover:border-emerald-400 dark:hover:border-emerald-600 bg-slate-50/70 dark:bg-slate-750/70 hover:bg-emerald-50/30 dark:hover:bg-emerald-950/20 cursor-pointer transition-all group"
                        >
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="font-bold text-emerald-700 dark:text-emerald-400 group-hover:underline flex items-center gap-1">
                              <span>
                                {language === "ar"
                                  ? `المادة ${cite.article_number}`
                                  : `Article ${cite.article_number}`}
                              </span>
                              <ExternalLink className="w-3 h-3 opacity-60 group-hover:opacity-100" />
                            </span>
                            {cite.is_repealed && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400 font-medium">
                                {language === "ar" ? "ملغاة" : "Repealed"}
                              </span>
                            )}
                          </div>
                          <p className="text-[11px] text-slate-600 dark:text-slate-300 line-clamp-2 leading-relaxed">
                            {cite.text_ar_snippet || cite.text_en_snippet}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Timestamp */}
              <span className="text-[10px] text-slate-400 mt-1 px-1">
                {msg.timestamp}
              </span>
            </div>
          </div>
        );
      })}

      {isLoading && (
        <div className="flex items-center gap-3 text-xs py-2.5 px-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm text-slate-600 dark:text-slate-300 rounded-2xl border border-emerald-200/60 dark:border-emerald-800/50 shadow-sm w-fit animate-pulse">
          <div className="flex gap-1.5 items-center">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-bounce" />
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-bounce [animation-delay:0.2s]" />
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-bounce [animation-delay:0.4s]" />
          </div>
          <span className="font-medium text-emerald-800 dark:text-emerald-300 flex items-center gap-1.5">
            <Scale className="w-3.5 h-3.5" />
            <span>
              {language === "ar"
                ? "جاري استرجاع السند القانوني والتحقق من المواد..."
                : "Retrieving legal statutes and verifying grounding..."}
            </span>
          </span>
        </div>
      )}
    </div>
  );
};
