import React from "react";
import { BookMarked, AlertTriangle, ExternalLink, X } from "lucide-react";
import { CitationItem } from "../types";

export interface CitationCardProps {
  citation: CitationItem;
  isOpen?: boolean;
  onClose: () => void;
  onViewInBrowser?: (articleNumber: number) => void;
  lang?: "ar" | "en";
  language?: "ar" | "en";
}

export const CitationCard: React.FC<CitationCardProps> = ({
  citation,
  isOpen = true,
  onClose,
  onViewInBrowser,
  lang,
  language,
}) => {
  const currentLang = language || lang || "ar";
  const isAr = currentLang === "ar";
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl bg-white dark:bg-slate-900 rounded-2xl shadow-2xl border border-slate-200 dark:border-slate-800 p-6 overflow-hidden max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 flex items-center justify-center">
              <BookMarked className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-base text-slate-900 dark:text-white flex items-center gap-2">
                <span>{isAr ? `المادة ${citation.article_number}` : `Article ${citation.article_number}`}</span>
                {citation.is_repealed && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-400 font-semibold border border-amber-200 dark:border-amber-800 flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3" />
                    <span>{isAr ? "مادة ملغاة" : "Repealed"}</span>
                  </span>
                )}
                {!citation.is_grounded && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-rose-100 dark:bg-rose-950 text-rose-700 dark:text-rose-400 font-semibold border border-rose-200 dark:border-rose-800">
                    {isAr ? "غير مؤكدة في السياق" : "Ungrounded in Context"}
                  </span>
                )}
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {citation.citation}
                {citation.source_page > 0 && ` • ${isAr ? "صفحة" : "Page"} ${citation.source_page}`}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body content */}
        <div className="py-4 space-y-4 overflow-y-auto flex-1">
          {/* Arabic text block */}
          <div className="bg-slate-50 dark:bg-slate-850/60 p-4 rounded-xl border border-slate-200 dark:border-slate-800">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block mb-1">
              النص العربي (القانون المدني المصري 1948)
            </span>
            <p className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed font-arabic whitespace-pre-wrap">
              {citation.text_ar_snippet || (isAr ? "لا يتوفر نص مقتطف لهذه المادة." : "No snippet available.")}
            </p>
          </div>

          {/* English translation block */}
          <div className="bg-slate-50 dark:bg-slate-850/60 p-4 rounded-xl border border-slate-200 dark:border-slate-800" dir="ltr">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block mb-1">
              Official English Translation
            </span>
            <p className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed font-sans whitespace-pre-wrap">
              {citation.text_en_snippet || "No English translation snippet available."}
            </p>
          </div>
        </div>

        {/* Footer actions */}
        <div className="pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
          {onViewInBrowser ? (
            <button
              type="button"
              onClick={() => onViewInBrowser(citation.article_number)}
              className="flex items-center gap-1.5 text-xs font-semibold text-emerald-600 dark:text-emerald-400 hover:underline"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              <span>
                {isAr
                  ? `عرض المادة ${citation.article_number} في المتصفح الكامل`
                  : `Open Article ${citation.article_number} in Full Browser`}
              </span>
            </button>
          ) : <div />}
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-xl text-xs font-semibold transition"
          >
            {isAr ? "إغلاق" : "Close"}
          </button>
        </div>
      </div>
    </div>
  );
};
