# Sanad (سند) — Arabic Legal Q&A System

**Sanad (سند)** is an Arabic legal Q&A system grounded on the **Egyptian Civil Code (القانون المدني المصري, Law No. 131 of 1948)**.

A user asks a question in Arabic or English; the system retrieves relevant articles using hybrid dense + sparse (BM25) search, checks grounding, and answers strictly based on retrieved articles with explicit citations (e.g. `[المادة 147]`).
