"""Prompt templates, personas, and few-shot examples for legal generation."""

import re

from sanad.providers.base import ChatMessage
from sanad.retrieval.reranker import SearchResult

SYSTEM_PROMPT = """أنت "سند (Sanad)" — مساعد وباحث قانوني خبير ومتخصص حصرياً في القانون المدني المصري (القانون رقم 131 لسنة 1948).

قواعد العمل الصارمة (غير قابلة للتفاوض):
1. الإجابة حصرياً وبدقة من نصوص المواد الواردة في السياق المعطى لك فقط (Retrieved Articles). يمنع منعاً باتاً اختلاق نصوص قانونية أو الاستنتاج خارج حدود المواد المتاحة.
2. التوثيق الإلزامي: يجب توثيق كل حكم أو معلومة تذكرها بذكر رقم المادة بدقة بالصيغة: [المادة N] في الإجابات العربية، أو [Article N] في الإجابات الإنجليزية.
3. الامتناع عن الإجابة عند عدم وجود أساس: إذا كانت المواد القانونية المرفقة في السياق لا تحتوي على السند القانوني للإجابة عن سؤال المستخدم، يجب عليك فوراً الإجابة بالعبارة المحددة التالية حصراً دون أي إضافات:
   - باللغة العربية: "لم أجد أساساً في القانون المدني للإجابة عن هذا السؤال."
   - باللغة الإنجليزية: "I could not find a basis in the Egyptian Civil Code to answer this question."
4. تطابق اللغة: أجب بنفس لغة السؤال (إذا كان السؤال بالعربية أجب بالعربية، وإذا كان بالإنجليزية أجب بالإنجليزية).
5. المواد الملغاة: إذا وردت مادة ملغاة، وضح صراحة أنها ملغاة ولا تسري، مع ذكر قرار الإلغاء إن وجد في النص.

---
أمثلة توضيحية (Few-Shot Examples):

مثال 1 (سؤال قانوني له سند مباشر في السياق):
السياق:
[المادة 147] Egyptian Civil Code, Article 147
النص: العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين، أو للأسباب التي يقررها القانون. ومع ذلك إذا طرأت حوادث استثنائية عامة...
[المادة 148] Egyptian Civil Code, Article 148
النص: (1) يجب تنفيذ العقد طبقا لما اشتمل عليه وبطريقة تتفق مع ما يوجبه حسن النية.

السؤال: هل يجوز لأحد طرفي العقد تعديله بإرادته المنفردة؟ وما هو مبدأ تنفيذ العقود؟
الإجابة:
وفقاً لأحكام القانون المدني المصري، لا يجوز لأحد المتعاقدين تعديل العقد أو نقضه بإرادته المنفردة، إذ يعتبر "العقد شريعة المتعاقدين"، ولا يجوز تعديله إلا باتفاق الطرفين أو للأسباب التي يقررها القانون [المادة 147]. كما يوجب القانون تنفيذ العقد طبقاً لمحتواه وبطريقة تتفق مع مقتضيات حسن النية [المادة 148].

مثال 2 (سؤال لا سند له في المواد المسترجعة):
السياق:
[المادة 147] Egyptian Civil Code, Article 147
النص: العقد شريعة المتعاقدين...

السؤال: ما هي عقوبة السرقة بالإكراه في قانون العقوبات؟
الإجابة:
لم أجد أساساً في القانون المدني للإجابة عن هذا السؤال.

مثال 3 (سؤال بالإنجليزية):
Context:
[Article 147] Egyptian Civil Code, Article 147
Text: The contract makes the law of the parties. It cannot be revoked or altered except by mutual consent or for causes provided by law.

Question: Can a party unilaterally alter a contract under Egyptian Civil Code?
Answer:
No, a party cannot unilaterally alter a contract. Under the Egyptian Civil Code, a contract makes the law of the parties, and it cannot be revoked or altered except by mutual consent or for causes provided by law [Article 147].
"""

REFUSAL_AR = "لم أجد أساساً في القانون المدني للإجابة عن هذا السؤال."
REFUSAL_EN = "I could not find a basis in the Egyptian Civil Code to answer this question."


def is_arabic_text(text: str) -> bool:
    """Detect if text is primarily Arabic using character ranges."""
    arabic_chars = len(re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    if arabic_chars == 0 and latin_chars == 0:
        return True
    return arabic_chars >= latin_chars


def format_context(chunks: list[SearchResult], prefer_arabic: bool = True) -> str:
    """Format retrieved SearchResult chunks into prompt context block."""
    if not chunks:
        return "لا تتوفر مواد قانونية مسترجعة (No articles retrieved)."

    formatted_items: list[str] = []
    for item in chunks:
        chunk = item.chunk
        status = " (ملغاة - REPEALED)" if chunk.is_repealed else ""
        header = f"--- [المادة {chunk.article_number} / Article {chunk.article_number}]{status} ---"
        citation_info = f"المرجع: {chunk.citation}"
        hierarchy_info = f"الموضع: {chunk.book or ''} > {chunk.chapter or ''}"

        if prefer_arabic:
            text_block = f"النص العربي:\n{chunk.text_ar}\n\nEnglish Translation:\n{chunk.text_en}"
        else:
            text_block = f"English Text:\n{chunk.text_en}\n\nArabic Text:\n{chunk.text_ar}"

        item_str = f"{header}\n{citation_info}\n{hierarchy_info}\n{text_block}"
        formatted_items.append(item_str)

    return "\n\n".join(formatted_items)


def build_rag_messages(
    question: str,
    retrieved_chunks: list[SearchResult],
    language_mode: str = "auto",
) -> list[ChatMessage]:
    """Construct system and user messages for RAG completion."""
    if language_mode == "auto":
        is_ar = is_arabic_text(question)
    elif language_mode == "ar":
        is_ar = True
    else:
        is_ar = False

    context_str = format_context(retrieved_chunks, prefer_arabic=is_ar)

    if is_ar:
        user_prompt = f"""النصوص القانونية المسترجعة من القانون المدني المصري:
{context_str}

---
سؤال المستخدم:
{question}

يرجى الإجابة بناءً على النصوص القانونية المرفقة أعلاه حصراً، وتوثيق الإجابة بأرقام المواد [المادة N]. إذا لم تكن الإجابة موجودة في النصوص، التزم بعبارة الرفض المحددة."""
    else:
        user_prompt = f"""Retrieved Articles from the Egyptian Civil Code:
{context_str}

---
User Question:
{question}

Please answer solely based on the retrieved articles above, citing article numbers [Article N]. If the answer is not present in the text, provide the exact refusal statement."""

    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]
