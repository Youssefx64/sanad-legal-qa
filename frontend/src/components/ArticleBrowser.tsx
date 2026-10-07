import React, { useState, useEffect } from "react";
import { ArticleRecord, CorpusStatsResponse } from "../types";
import { fetchArticle, fetchCorpusStats } from "../api";
import {
  BookOpen,
  Search,
  ChevronRight,
  ChevronLeft,
  Copy,
  Check,
  AlertTriangle,
  FileText,
  Bookmark,
  Layers,
} from "lucide-react";

interface ArticleBrowserProps {
  initialArticleNumber?: number;
  language: "ar" | "en";
  onArticleChange?: (num: number) => void;
}

export const ArticleBrowser: React.FC<ArticleBrowserProps> = ({
  initialArticleNumber = 1,
  language,
  onArticleChange,
}) => {
  const [currentNumber, setCurrentNumber] = useState<number>(initialArticleNumber);
  const [article, setArticle] = useState<ArticleRecord | null>(null);
  const [stats, setStats] = useState<CorpusStatsResponse | null>(null);
  const [searchInput, setSearchInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"bilingual" | "ar" | "en" | "raw">("bilingual");
  const [copied, setCopied] = useState<boolean>(false);

  useEffect(() => {
    fetchCorpusStats()
      .then(setStats)
      .catch((err) => console.error("Failed to fetch corpus stats:", err));
  }, []);

  useEffect(() => {
    if (initialArticleNumber && initialArticleNumber !== currentNumber) {
      setCurrentNumber(initialArticleNumber);
    }
  }, [initialArticleNumber]);

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    fetchArticle(currentNumber)
      .then((data) => {
        if (isMounted) {
          setArticle(data);
          setLoading(false);
          if (onArticleChange) onArticleChange(data.article_number);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || "Failed to load article");
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [currentNumber]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const parsed = parseInt(searchInput.trim(), 10);
    if (!isNaN(parsed) && parsed >= 1 && parsed <= 1149) {
      setCurrentNumber(parsed);
      setSearchInput("");
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex-1 flex flex-col md:flex-row overflow-hidden max-w-6xl w-full mx-auto p-4 gap-4">
      {/* Sidebar Controls */}
      <div className="w-full md:w-80 shrink-0 flex flex-col gap-3">
        {/* Search Card */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm">
          <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100 mb-3 flex items-center gap-2">
            <Search className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            <span>{language === "ar" ? "الانتقال إلى مادة" : "Jump to Article"}</span>
          </h3>

          <form onSubmit={handleSearchSubmit} className="flex gap-2 mb-3">
            <input
              type="number"
              min={1}
              max={1149}
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder={language === "ar" ? "رقم المادة (1 - 1149)..." : "Article # (1 - 1149)..."}
              className="w-full text-sm px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-850 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500/30"
            />
            <button
              type="submit"
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl text-sm font-medium transition-colors"
            >
              {language === "ar" ? "انتقال" : "Go"}
            </button>
          </form>

          {/* Quick Step Controls */}
          <div className="flex items-center justify-between pt-2 border-t border-slate-100 dark:border-slate-700/60 text-xs">
            <button
              type="button"
              disabled={currentNumber <= 1}
              onClick={() => setCurrentNumber((prev) => Math.max(1, prev - 1))}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-750 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-750 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              {language === "ar" ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
              <span>{language === "ar" ? "السابقة" : "Previous"}</span>
            </button>

            <span className="font-mono font-bold text-slate-700 dark:text-slate-300">
              {currentNumber} / 1149
            </span>

            <button
              type="button"
              disabled={currentNumber >= 1149}
              onClick={() => setCurrentNumber((prev) => Math.min(1149, prev + 1))}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-750 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-750 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <span>{language === "ar" ? "التالية" : "Next"}</span>
              {language === "ar" ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Corpus Quick Stats */}
        {stats && (
          <div className="bg-white dark:bg-slate-800 p-4 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm text-xs">
            <h4 className="font-bold text-slate-700 dark:text-slate-300 mb-2 flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
              <span>{language === "ar" ? "إحصاءات المدونة" : "Code Structure"}</span>
            </h4>
            <div className="space-y-1.5 text-slate-500 dark:text-slate-400">
              <div className="flex justify-between">
                <span>{language === "ar" ? "إجمالي المواد:" : "Total Articles:"}</span>
                <span className="font-semibold text-slate-800 dark:text-slate-200">{stats.total_articles}</span>
              </div>
              <div className="flex justify-between">
                <span>{language === "ar" ? "المواد السارية:" : "Active Articles:"}</span>
                <span className="font-semibold text-emerald-600 dark:text-emerald-400">{stats.active_articles}</span>
              </div>
              <div className="flex justify-between">
                <span>{language === "ar" ? "المواد الملغاة:" : "Repealed Articles:"}</span>
                <span className="font-semibold text-rose-600 dark:text-rose-400">{stats.repealed_articles}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main Article Display Card */}
      <div className="flex-1 bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm flex flex-col overflow-hidden">
        {loading ? (
          <div className="flex-1 flex items-center justify-center p-8 text-slate-400">
            <div className="flex flex-col items-center gap-3">
              <BookOpen className="w-8 h-8 animate-pulse text-emerald-600" />
              <p className="text-xs">{language === "ar" ? "جاري تحميل نص المادة..." : "Loading article..."}</p>
            </div>
          </div>
        ) : error ? (
          <div className="flex-1 flex items-center justify-center p-8 text-rose-500">
            <div className="flex flex-col items-center gap-2">
              <AlertTriangle className="w-8 h-8" />
              <p className="text-sm font-semibold">{error}</p>
            </div>
          </div>
        ) : article ? (
          <div className="flex-1 flex flex-col overflow-y-auto p-6">
            {/* Article Header */}
            <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-100 dark:border-slate-700">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 font-bold text-lg border border-emerald-200 dark:border-emerald-800">
                  {article.article_number}
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">
                    {language === "ar"
                      ? `المادة ${article.article_number}`
                      : `Article ${article.article_number}`}
                  </h2>
                  <p className="text-xs text-slate-400 dark:text-slate-500">
                    {article.citation} • {language === "ar" ? `صفحة ${article.source_page}` : `Page ${article.source_page}`}
                  </p>
                </div>
              </div>

              {/* Status and Action Badges */}
              <div className="flex items-center gap-2">
                {article.is_repealed && (
                  <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-400 border border-rose-200 dark:border-rose-800">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span>{language === "ar" ? "مادة ملغاة بموجب قانون" : "Repealed Article"}</span>
                  </span>
                )}

                <button
                  type="button"
                  onClick={() => handleCopy(article.text_ar + "\n\n" + article.text_en)}
                  className="p-2 rounded-xl text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
                  title={language === "ar" ? "نسخ نص المادة" : "Copy text"}
                >
                  {copied ? <Check className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Hierarchical Breadcrumbs */}
            <div className="py-3 flex flex-wrap gap-2 text-xs text-slate-500 dark:text-slate-400">
              {article.book && (
                <span className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-750 font-medium">
                  {article.book}
                </span>
              )}
              {article.chapter && (
                <span className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-750">
                  {article.chapter}
                </span>
              )}
              {article.section && (
                <span className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-750">
                  {article.section}
                </span>
              )}
              {article.topic && (
                <span className="px-2.5 py-1 rounded-md bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
                  {article.topic}
                </span>
              )}
            </div>

            {/* Language Tabs */}
            <div className="flex border-b border-slate-100 dark:border-slate-750 mb-4 text-xs font-semibold">
              <button
                type="button"
                onClick={() => setActiveTab("bilingual")}
                className={`py-2 px-3 border-b-2 transition-colors ${
                  activeTab === "bilingual"
                    ? "border-emerald-600 text-emerald-600 dark:text-emerald-400"
                    : "border-transparent text-slate-500 hover:text-slate-700"
                }`}
              >
                {language === "ar" ? "عرض ثنائي اللغة" : "Bilingual View"}
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("ar")}
                className={`py-2 px-3 border-b-2 transition-colors ${
                  activeTab === "ar"
                    ? "border-emerald-600 text-emerald-600 dark:text-emerald-400"
                    : "border-transparent text-slate-500 hover:text-slate-700"
                }`}
              >
                {language === "ar" ? "النص العربي فقط" : "Arabic Only"}
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("en")}
                className={`py-2 px-3 border-b-2 transition-colors ${
                  activeTab === "en"
                    ? "border-emerald-600 text-emerald-600 dark:text-emerald-400"
                    : "border-transparent text-slate-500 hover:text-slate-700"
                }`}
              >
                {language === "ar" ? "النص الإنجليزي فقط" : "English Only"}
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("raw")}
                className={`py-2 px-3 border-b-2 transition-colors ${
                  activeTab === "raw"
                    ? "border-emerald-600 text-emerald-600 dark:text-emerald-400"
                    : "border-transparent text-slate-500 hover:text-slate-700"
                }`}
              >
                {language === "ar" ? "النص الخام المستخرج" : "Raw Extracted Arabic"}
              </button>
            </div>

            {/* Article Content */}
            <div className="flex-1 space-y-4">
              {(activeTab === "bilingual" || activeTab === "ar") && (
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-850/60 border border-slate-200/80 dark:border-slate-750">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-slate-600 dark:text-slate-400 mb-2">
                    <FileText className="w-3.5 h-3.5 text-emerald-600" />
                    <span>النص العربي (القانون المدني المصري):</span>
                  </div>
                  <p
                    className="text-base text-slate-900 dark:text-slate-100 leading-relaxed font-arabic whitespace-pre-wrap"
                    dir="rtl"
                  >
                    {article.text_ar}
                  </p>
                </div>
              )}

              {(activeTab === "bilingual" || activeTab === "en") && (
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-850/60 border border-slate-200/80 dark:border-slate-750">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-slate-600 dark:text-slate-400 mb-2">
                    <FileText className="w-3.5 h-3.5 text-emerald-600" />
                    <span>English Translation:</span>
                  </div>
                  <p
                    className="text-sm text-slate-800 dark:text-slate-200 leading-relaxed whitespace-pre-wrap font-sans"
                    dir="ltr"
                  >
                    {article.text_en}
                  </p>
                </div>
              )}

              {activeTab === "raw" && (
                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-850/60 border border-slate-200/80 dark:border-slate-750">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-amber-600 dark:text-amber-400 mb-2">
                    <FileText className="w-3.5 h-3.5" />
                    <span>النص الخام بدون معالجة (قبل التوحيد الإملائي):</span>
                  </div>
                  <pre
                    className="text-xs text-slate-700 dark:text-slate-300 font-mono whitespace-pre-wrap leading-relaxed"
                    dir="rtl"
                  >
                    {article.text_ar_raw}
                  </pre>
                </div>
              )}

              {/* Cross References */}
              {article.references && article.references.length > 0 && (
                <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-750">
                  <h4 className="text-xs font-bold text-slate-500 dark:text-slate-400 mb-2 flex items-center gap-1">
                    <Bookmark className="w-3.5 h-3.5" />
                    <span>
                      {language === "ar" ? "إحالات إلى مواد أخرى في نص المادة:" : "Statutory Cross-References:"}
                    </span>
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {article.references.map((refNum) => (
                      <button
                        key={refNum}
                        type="button"
                        onClick={() => setCurrentNumber(refNum)}
                        className="px-2.5 py-1 rounded-lg bg-emerald-50 hover:bg-emerald-100 dark:bg-emerald-950/60 dark:hover:bg-emerald-900/60 text-emerald-800 dark:text-emerald-300 font-semibold text-xs border border-emerald-200 dark:border-emerald-800 transition-colors"
                      >
                        {language === "ar" ? `المادة ${refNum}` : `Article ${refNum}`}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
};
