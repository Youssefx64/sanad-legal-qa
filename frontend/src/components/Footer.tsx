import React from "react";
import { Scale, ShieldAlert } from "lucide-react";

interface FooterProps {
  language: "ar" | "en";
}

export const Footer: React.FC<FooterProps> = ({ language }) => {
  return (
    <footer className="border-t border-slate-200/80 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm py-3 px-4 text-xs">
      <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-slate-500 dark:text-slate-400">
        <div className="flex items-center gap-2 text-center sm:text-start">
          <ShieldAlert className="w-4 h-4 text-amber-500 shrink-0" />
          <p>
            {language === "ar"
              ? "إخلاء مسؤولية: سند نظام بحث وتأصيل قانوني تجريبي، ولا يغني عن استشارة محامٍ مقيد قانوناً."
              : "Disclaimer: Sanad is an experimental legal Q&A tool and does not constitute formal legal advice."}
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0 font-medium text-[11px] text-slate-400">
          <span className="flex items-center gap-1">
            <Scale className="w-3.5 h-3.5 text-emerald-600" />
            <span>القانون المدني المصري (1948)</span>
          </span>
          <span>•</span>
          <span>v1.0.0</span>
        </div>
      </div>
    </footer>
  );
};
