"""Locust load test simulating 50 concurrent users on Sanad Legal Q&A API."""

import random

from locust import HttpUser, between, task

SAMPLE_QUESTIONS = [
    {
        "question": "ما هي القوة الملزمة للعقد وما هو أثر الظروف الطارئة وفقاً للمادة 147؟",
        "language": "ar",
    },
    {
        "question": "كيف يتم تنفيذ العقد بحسن نية وما الذي يشمله العقد وفقاً للمادة 148؟",
        "language": "ar",
    },
    {
        "question": "ما هي القاعدة العامة في المسؤولية التقصيرية عن العمل غير المشروع وفقاً للمادة 163؟",
        "language": "ar",
    },
    {"question": "ما هي سلطات المالك على ملكه ونطاق حق الملكية وفقاً للمادة 802؟", "language": "ar"},
    {
        "question": "ما هو تعريف عقد الإيجار وما هي عناصره الأساسية وفقاً للمادة 558؟",
        "language": "ar",
    },
    {"question": "What is the binding force of contract under Article 147?", "language": "en"},
    {"question": "What are the elements of tort liability under Article 163?", "language": "en"},
    {"question": "What is the statutory limitation period under Article 374?", "language": "en"},
]


class SanadLegalQAUser(HttpUser):
    """Simulated legal researcher querying Sanad API."""

    wait_time = between(0.5, 2.0)

    @task(5)
    def ask_legal_question(self) -> None:
        """Query standard POST /ask endpoint."""
        q = random.choice(SAMPLE_QUESTIONS)
        payload = {
            "question": q["question"],
            "top_k": 5,
            "language": q["language"],
        }
        with self.client.post(
            "/ask",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="POST /ask",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "answer" in data and "citations" in data:
                    resp.success()
                else:
                    resp.failure("Missing answer or citations in payload")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text}")

    @task(2)
    def browse_article(self) -> None:
        """Fetch article by number."""
        article_num = random.randint(1, 1149)
        self.client.get(
            f"/articles/{article_num}",
            name="GET /articles/{number}",
        )

    @task(1)
    def get_models_catalog(self) -> None:
        """Fetch models catalog."""
        self.client.get("/models", name="GET /models")

    @task(1)
    def get_corpus_stats(self) -> None:
        """Fetch corpus stats."""
        self.client.get("/corpus/stats", name="GET /corpus/stats")
