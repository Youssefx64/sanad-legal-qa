"""Parser to build structured ArticleRecord instances from extracted PDF pages."""

import re
from pathlib import Path
from typing import Any

from sanad.corpus.arabic_text import (
    clean_arabic_raw,
    extract_article_references,
    normalize_arabic,
)
from sanad.corpus.pdf_extract import PageData, extract_pdf_pages
from sanad.corpus.schema import ArticleRecord, IssuanceLawArticle

# Repealed article ranges
REPEALED_54_80 = set(range(54, 81))
REPEALED_389_417 = set(range(389, 418))

REPEAL_54_80_NOTE_AR = (
    "ألغيت المواد من 54 إلى 80 بالقرار الجمهوري بالقانون رقم 384 لسنة 1956 "
    "(الوقائع المصرية عدد 88 مكرر (ج) في 3/11/1956) ثم ألغي القانون بالقرار الجمهوري "
    "رقم 32 لسنة 1964 (الجريدة الرسمية عدد 37 في 12/3/1964)."
)
REPEAL_54_80_NOTE_EN = (
    "Articles 54-80 have been repealed by Presidential Decree Law No. 384 of 1956 "
    "and Presidential Decree No. 32 of 1964."
)

REPEAL_389_417_NOTE_AR = (
    "ألغيت المواد من 389 إلى 417 بموجب قانون الإثبات في المواد المدنية والتجارية "
    "رقم 25 لسنة 1968 (الجريدة الرسمية عدد 22 في 30/5/1968)."
)
REPEAL_389_417_NOTE_EN = (
    "Articles 389-417 have been repealed by the Law of Evidence in Civil and Commercial "
    "Matters (Law No. 25 of 1968)."
)


def _clean_article_headers(text_en: str, text_ar: str) -> tuple[str, str]:
    """Strip leading 'Article N' and 'مادة N' markers from article bodies."""
    cleaned_en = re.sub(
        r"^(?:Article|rticle)\s*\d+[\s\.:\-\)]*", "", text_en.strip(), flags=re.IGNORECASE
    ).strip()
    cleaned_ar = re.sub(
        r"^[\s\(\[]*م\s*ا\s*د\s*ة[\s\(\[\d\u0660-\u0669\-\.\:\)]*", "", text_ar.strip()
    ).strip()
    cleaned_ar = re.sub(r"^[\s\(\[]*[\d\u0660-\u0669]+[\s\)\.\:\-]*", "", cleaned_ar).strip()
    return cleaned_en, cleaned_ar


class CorpusParser:
    """Orchestrates extraction and parsing of all Egyptian Civil Code articles."""

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self.pages: list[PageData] = []
        self.issuance_law: list[IssuanceLawArticle] = []
        self.articles: dict[int, ArticleRecord] = {}

    def parse(self) -> tuple[list[ArticleRecord], list[IssuanceLawArticle]]:
        """Parse the source PDF and return contiguous 1..1149 articles + issuance law."""
        self.pages = extract_pdf_pages(self.pdf_path)
        self._parse_issuance_law()
        self._parse_code_articles()

        sorted_articles = [self.articles[i] for i in sorted(self.articles.keys())]
        return sorted_articles, self.issuance_law

    def _parse_issuance_law(self) -> None:
        """Extract articles 1 and 2 of the Issuance Law (قانون الإصدار) from page 1."""
        art1_ar = (
            "يلغي القانون المدني المعمول به أمام المحاكم الوطنية والصادر في 28 أكتوبر سنة 1883 "
            "والقانون المدني المعمول به أمام المحاكم المختلطة والصادر في 28 يونيو سنة 1875 "
            "ويستعاض عنهما بالقانون المدني المرافق لهذا القانون."
        )
        art1_en = (
            "The Civil Code in force before the national courts promulgated on October 28, 1883 "
            "and the Civil Code in force before the mixed courts promulgated on June 28, 1875 "
            "are repealed and replaced by the Civil Code annexed to the present law."
        )
        art2_ar = "على وزير العدل تنفيذ هذا القانون ويعمل به ابتداء من 15 أكتوبر سنة 1949."
        art2_en = (
            "The Minister of Justice is charged with the execution of this law, which shall come "
            "into force on October 15, 1949."
        )

        self.issuance_law = [
            IssuanceLawArticle(
                article_number=1,
                text_ar=art1_ar,
                text_en=art1_en,
                source_page=1,
                citation="Egyptian Civil Code, Issuance Law Article 1",
            ),
            IssuanceLawArticle(
                article_number=2,
                text_ar=art2_ar,
                text_en=art2_en,
                source_page=1,
                citation="Egyptian Civil Code, Issuance Law Article 2",
            ),
        ]

    def _parse_code_articles(self) -> None:
        """Parse all articles 1..1149 of the Egyptian Civil Code."""
        # 1. Identify all valid sequential English article start spans
        en_spans: list[tuple[int, int, float]] = []

        for p_idx, page in enumerate(self.pages):
            ar_mada_ys = [ln.y0 for ln in page.ar_lines if re.search(r"م\s*ا\s*د\s*ة", ln.text)]
            for line in page.en_lines:
                m = re.match(r"^(?:Article|rticle)\s*(\d+)\b", line.text, re.IGNORECASE)
                if not m:
                    continue
                num = int(m.group(1))
                rem = line.text[m.end() :].strip()
                has_words = len(rem.split()) > 1
                has_near_mada = any(abs(l_y - line.y0) < 30 for l_y in ar_mada_ys)

                # Skip false positives: citations in running sentences without nearby Arabic header
                if has_words and not has_near_mada:
                    continue
                if rem.startswith(".") and not has_near_mada:
                    continue

                if not en_spans:
                    if num == 1:
                        en_spans.append((num, p_idx, line.y0))
                else:
                    last_num = en_spans[-1][0]
                    expected_max = last_num + 35 if last_num in [53, 54, 388] else last_num + 5
                    if num > last_num and num <= expected_max:
                        en_spans.append((num, p_idx, line.y0))

        # 2. Track hierarchy across pages
        hierarchy_by_page = self._build_page_hierarchy()

        # 3. Process every article from spans
        en_lines_by_page = [p.en_lines for p in self.pages]
        ar_lines_by_page = [p.ar_lines for p in self.pages]

        for i, (art_num, sp, sy) in enumerate(en_spans):
            if i + 1 < len(en_spans):
                _, ep, ey = en_spans[i + 1]
            else:
                ep, ey = len(self.pages) - 1, 99999.0

            en_raw_lines = self._get_lines_in_span(en_lines_by_page, sp, sy, ep, ey, offset=0.0)
            ar_raw_lines = self._get_lines_in_span(ar_lines_by_page, sp, sy, ep, ey, offset=-2.0)

            en_raw = " ".join(en_raw_lines)
            ar_raw = " ".join(ar_raw_lines)

            cleaned_en, cleaned_ar = _clean_article_headers(en_raw, ar_raw)

            cleaned_ar_raw = clean_arabic_raw(cleaned_ar)
            norm_ar = normalize_arabic(cleaned_ar_raw)

            hier = hierarchy_by_page.get(sp + 1, {})
            refs = extract_article_references(cleaned_ar, cleaned_en, current_article=art_num)

            self.articles[art_num] = ArticleRecord(
                article_number=art_num,
                book=hier.get("book"),
                part=hier.get("part"),
                chapter=hier.get("chapter"),
                section=hier.get("section"),
                topic=hier.get("topic"),
                text_ar=norm_ar if norm_ar else cleaned_ar_raw,
                text_ar_raw=cleaned_ar_raw,
                text_en=cleaned_en,
                is_repealed=False,
                source_page=sp + 1,
                citation=f"Egyptian Civil Code, Article {art_num}",
                references=refs,
            )

        # 4. Handle repealed ranges
        self._populate_repealed_ranges(hierarchy_by_page)

        # 5. Handle special cases (Article 1021 and 1022 separation)
        self._reconcile_article_1022()

    def _get_lines_in_span(
        self,
        lines_by_page: list[list[Any]],
        start_p: int,
        start_y: float,
        end_p: int,
        end_y: float,
        offset: float = 0.0,
    ) -> list[str]:
        """Extract text lines within a page and vertical boundary."""
        lines: list[str] = []
        for p in range(start_p, end_p + 1):
            for ln in lines_by_page[p]:
                ly = ln.y0 + offset
                if p == start_p and ly < start_y - 10:
                    continue
                if p == end_p and ly >= end_y - 4:
                    continue
                lines.append(ln.text)
        return lines

    def _build_page_hierarchy(self) -> dict[int, dict[str, str]]:
        """Scan headings across pages to track active hierarchy."""
        hierarchy: dict[int, dict[str, str]] = {}
        current_book = "General Provisions (باب تمهيدي)"
        current_part = "General Provisions"
        current_chapter = "General Provisions"
        current_section = "Law and its Application"
        current_topic = None

        for p_idx, page in enumerate(self.pages):
            page_num = p_idx + 1

            for ln in page.en_lines:
                t = ln.text.strip()
                if re.match(r"^BOOK\s+[IVXLCDM]+", t, re.I):
                    current_book = t
                elif re.match(r"^(?:FIRST|SECOND|THIRD|FOURTH)\s+PART", t, re.I):
                    current_part = t
                elif re.match(r"^CHAPTER\s+[IVXLCDM]+", t, re.I):
                    current_chapter = t
                elif re.match(r"^SECTION\s+[IVXLCDM]+", t, re.I):
                    current_section = t
                elif re.match(r"^\d+\.\s+[A-Za-z]", t):
                    current_topic = t

            hierarchy[page_num] = {
                "book": current_book,
                "part": current_part,
                "chapter": current_chapter,
                "section": current_section,
                "topic": current_topic or "",
            }

        return hierarchy

    def _populate_repealed_ranges(self, hierarchy_by_page: dict[int, dict[str, str]]) -> None:
        """Ensure repealed articles 54-80 and 389-417 exist with appropriate flags."""
        # 54-80
        hier_54 = hierarchy_by_page.get(7, {})
        for num in range(54, 81):
            self.articles[num] = ArticleRecord(
                article_number=num,
                book=hier_54.get("book", "General Provisions"),
                part=hier_54.get("part", "Juristic Persons"),
                chapter=hier_54.get("chapter", "Classification of Things"),
                section=hier_54.get("section", "Associations"),
                topic="Repealed Articles",
                text_ar=normalize_arabic(REPEAL_54_80_NOTE_AR),
                text_ar_raw=REPEAL_54_80_NOTE_AR,
                text_en=REPEAL_54_80_NOTE_EN,
                is_repealed=True,
                source_page=7,
                citation=f"Egyptian Civil Code, Article {num} (Repealed)",
                references=[],
            )

        # 389-417
        hier_389 = hierarchy_by_page.get(53, {})
        for num in range(389, 418):
            self.articles[num] = ArticleRecord(
                article_number=num,
                book=hier_389.get("book", "Obligations Generally"),
                part=hier_389.get("part", "Obligations or Personal Rights"),
                chapter="Proof of Obligations (الباب السادس: إثبات الالتزام)",
                section="Proof (Repealed)",
                topic="Repealed Articles",
                text_ar=normalize_arabic(REPEAL_389_417_NOTE_AR),
                text_ar_raw=REPEAL_389_417_NOTE_AR,
                text_en=REPEAL_389_417_NOTE_EN,
                is_repealed=True,
                source_page=53,
                citation=f"Egyptian Civil Code, Article {num} (Repealed)",
                references=[],
            )

    def _reconcile_article_1022(self) -> None:
        """Handle omitted Arabic marker 1022 between paragraphs 1 and 2 of 1021."""
        if 1021 not in self.articles or 1022 not in self.articles:
            return

        art1021 = self.articles[1021]
        art1022 = self.articles[1022]

        # Arabic text of 1021 contains paragraphs (1), (2), (3)
        raw_ar = art1021.text_ar_raw
        if re.search(r"[\(\[]\s*[٢2]\s*[\(\)\]]", raw_ar):
            parts = re.split(r"[\(\[]\s*[٢2]\s*[\(\)\]]", raw_ar, maxsplit=1)
            if len(parts) == 2:
                p1_ar = parts[0].strip()
                p2_ar = parts[1].strip()

                # Update 1021
                art1021.text_ar_raw = clean_arabic_raw(p1_ar)
                art1021.text_ar = normalize_arabic(art1021.text_ar_raw)

                # Update 1022
                art1022.text_ar_raw = clean_arabic_raw(p2_ar)
                art1022.text_ar = normalize_arabic(art1022.text_ar_raw)


def parse_civil_code_corpus(
    pdf_path: str | Path,
) -> tuple[list[ArticleRecord], list[IssuanceLawArticle]]:
    """Convenience function to parse the entire corpus."""
    parser = CorpusParser(pdf_path)
    return parser.parse()
