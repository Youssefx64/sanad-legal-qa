import React from "react";
import { Cpu, Layers } from "lucide-react";
import { ModelsResponse } from "../types";

interface ModelSelectorProps {
  models: ModelsResponse | null;
  selectedChatModel: string;
  selectedEmbeddingModel: string;
  onChangeChatModel: (id: string) => void;
  onChangeEmbeddingModel: (id: string) => void;
  lang: "ar" | "en";
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedChatModel,
  selectedEmbeddingModel,
  onChangeChatModel,
  onChangeEmbeddingModel,
  lang,
}) => {
  const isAr = lang === "ar";

  if (!models) {
    return (
      <div className="flex items-center gap-2 py-2 text-xs text-slate-400">
        <div className="w-3 h-3 rounded-full border-2 border-brand-500 border-t-transparent animate-spin" />
        <span>{isAr ? "جاري تحميل النماذج..." : "Loading model catalog..."}</span>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-3 py-2 px-3 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-xs text-xs">
      {/* Chat Model Selector */}
      <div className="flex items-center gap-1.5 flex-1 min-w-[200px]">
        <Cpu className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400 shrink-0" />
        <span className="font-medium text-slate-500 dark:text-slate-400 shrink-0">
          {isAr ? "نموذج الإجابة:" : "Chat Model:"}
        </span>
        <select
          value={selectedChatModel}
          onChange={(e) => onChangeChatModel(e.target.value)}
          className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1 text-slate-800 dark:text-slate-200 focus:outline-hidden focus:ring-1 focus:ring-brand-500 w-full"
        >
          {models.chat_models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label} ({m.provider})
            </option>
          ))}
        </select>
      </div>

      {/* Embedding Model Selector */}
      <div className="flex items-center gap-1.5 flex-1 min-w-[200px]">
        <Layers className="w-3.5 h-3.5 text-brand-600 dark:text-brand-400 shrink-0" />
        <span className="font-medium text-slate-500 dark:text-slate-400 shrink-0">
          {isAr ? "نموذج التضمين:" : "Embedding:"}
        </span>
        <select
          value={selectedEmbeddingModel}
          onChange={(e) => onChangeEmbeddingModel(e.target.value)}
          className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1 text-slate-800 dark:text-slate-200 focus:outline-hidden focus:ring-1 focus:ring-brand-500 w-full"
        >
          {models.embedding_models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label} {m.dimension ? `(${m.dimension}d)` : ""}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};
