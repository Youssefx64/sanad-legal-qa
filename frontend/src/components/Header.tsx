import React from "react";
import { Scale, BookOpen, MessageSquare, Sun, Moon, Globe, Cpu, Layers } from "lucide-react";
import { ChatModelInfo, EmbeddingModelInfo } from "../types";

export interface HeaderProps {
  activeTab: "chat" | "browser";
  setActiveTab: (tab: "chat" | "browser") => void;
  language: "ar" | "en";
  setLanguage: (lang: "ar" | "en") => void;
  darkMode: boolean;
  setDarkMode: (dark: boolean) => void;
  chatModels: ChatModelInfo[];
  selectedChatModel: string;
  onSelectChatModel: (id: string) => void;
  embeddingModels: EmbeddingModelInfo[];
  selectedEmbeddingModel: string;
  onSelectEmbeddingModel: (id: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  language,
  setLanguage,
  darkMode,
  setDarkMode,
  chatModels,
  selectedChatModel,
  onSelectChatModel,
  embeddingModels,
  selectedEmbeddingModel,
  onSelectEmbeddingModel,
}) => {
  const isAr = language === "ar";

  return (
    <header className="sticky top-0 z-30 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 transition-colors shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-2.5 flex flex-col gap-2">
        <div className="flex items-center justify-between gap-4">
          {/* Brand / Logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-700 to-emerald-500 text-white flex items-center justify-center shadow-md shadow-emerald-600/20">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-xl tracking-tight text-slate-900 dark:text-white">
                  {isAr ? "سَنَد" : "SANAD"}
                </span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-950/80 text-emerald-800 dark:text-emerald-300 font-semibold border border-emerald-200 dark:border-emerald-800">
                  {isAr ? "القانون المدني 1948" : "Civil Code 1948"}
                </span>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 hidden sm:block">
                {isAr
                  ? "المساعد القانوني التوثيقي للقانون المدني المصري"
                  : "Egyptian Civil Code Legal AI Assistant"}
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex items-center bg-slate-100 dark:bg-slate-800 p-1 rounded-xl border border-slate-200 dark:border-slate-700">
            <button
              type="button"
              onClick={() => setActiveTab("chat")}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === "chat"
                  ? "bg-white dark:bg-slate-700 text-emerald-700 dark:text-emerald-300 shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>{isAr ? "المحادثة والاستشارة" : "Legal Chat"}</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("browser")}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeTab === "browser"
                  ? "bg-white dark:bg-slate-700 text-emerald-700 dark:text-emerald-300 shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>{isAr ? "تصفح نصوص القانون" : "Code Browser"}</span>
            </button>
          </div>

          {/* Settings & Controls */}
          <div className="flex items-center gap-2">
            {/* Language Switcher */}
            <button
              type="button"
              onClick={() => setLanguage(isAr ? "en" : "ar")}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg text-slate-700 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 transition"
              title={isAr ? "Switch to English" : "التبديل إلى العربية"}
            >
              <Globe className="w-3.5 h-3.5" />
              <span>{isAr ? "English" : "العربية"}</span>
            </button>

            {/* Dark Mode Toggle */}
            <button
              type="button"
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg transition"
              title={darkMode ? "Light Mode" : "Dark Mode"}
            >
              {darkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Model Bar */}
        <div className="flex flex-wrap items-center justify-end gap-3 text-xs pt-1 border-t border-slate-100 dark:border-slate-800/80">
          <div className="flex items-center gap-1 text-slate-500 dark:text-slate-400">
            <Cpu className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
            <span className="font-medium">{isAr ? "النموذج:" : "LLM:"}</span>
            <select
              value={selectedChatModel}
              onChange={(e) => onSelectChatModel(e.target.value)}
              className="bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-0.5 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-500 font-mono text-[11px]"
            >
              {chatModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label} ({m.provider})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1 text-slate-500 dark:text-slate-400">
            <Layers className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
            <span className="font-medium">{isAr ? "التضمين:" : "Embed:"}</span>
            <select
              value={selectedEmbeddingModel}
              onChange={(e) => onSelectEmbeddingModel(e.target.value)}
              className="bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-0.5 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-emerald-500 font-mono text-[11px]"
            >
              {embeddingModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label} {m.dimension ? `(${m.dimension}d)` : ""}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
    </header>
  );
};
