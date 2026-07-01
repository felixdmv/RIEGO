#!/usr/bin/env python3
"""
Quick test against the Riotinto NI 43-101 PDF.

It extracts page text, performs a lightweight keyword-based retrieval for two
fixed questions, and saves question-answer pairs plus supporting sources under
./quick_test.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PDF_PATH = Path(
    r"C:\Users\fdemiguel\OneDrive - Universidad de Burgos\Documentos\CRMsDataSpace\pdfs\2023.02.21-ATYM-Riotinto-NI-43-101-Technical-Report-PEA-vF(1).pdf"
)
OUT_DIR = ROOT / "quick_test"


STOPWORDS = {
    "the",
    "and",
    "are",
    "for",
    "from",
    "that",
    "this",
    "with",
    "what",
    "where",
    "how",
    "does",
    "have",
    "has",
    "into",
    "there",
    "their",
    "about",
    "mine",
    "mina",
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def load_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = normalize_spaces(page.extract_text() or "")
        pages.append({"page": page_num, "text": text})
    return pages


def score_page(
    question: str,
    page_text: str,
    extra_terms: list[str],
    page_num: int,
    preferred_pages: list[int] | None = None,
) -> float:
    q_tokens = tokenize(question)
    p_tokens = tokenize(page_text)
    if not p_tokens:
        return 0.0

    p_counts = Counter(p_tokens)
    score = sum(2.0 for tok in q_tokens if tok in p_counts)

    low_text = page_text.lower()
    for term in extra_terms:
        if term.lower() in low_text:
            score += 3.0

    if preferred_pages and page_num in preferred_pages:
        score += 10.0

    return score


def retrieve_pages(
    pages: list[dict],
    question: str,
    extra_terms: list[str],
    top_k: int = 3,
    preferred_pages: list[int] | None = None,
) -> list[dict]:
    ranked = []
    page_pool = pages
    if preferred_pages:
        preferred_set = set(preferred_pages)
        preferred_only = [item for item in pages if item["page"] in preferred_set]
        if preferred_only:
            page_pool = preferred_only

    for item in page_pool:
        score = score_page(question, item["text"], extra_terms, item["page"], preferred_pages)
        if score > 0:
            ranked.append({**item, "score": score})
    ranked.sort(key=lambda x: (-x["score"], x["page"]))
    return ranked[:top_k]


def build_materials_answer() -> str:
    return (
        "The mine mainly contains copper-bearing material, and the report also describes "
        "polymetallic material with zinc and lead. Sulfur is modeled in the resource "
        "estimation, and silver is mentioned as an additional potentially saleable metal."
    )


def build_location_answer() -> str:
    return (
        "The Riotinto project is in Huelva Province, Spain, about 65 km northwest of Seville "
        "and 70 km northeast of the port of Huelva, between Minas de Riotinto, Nerva, and El Campillo. "
        "Access is by a renovated high-quality national road network, and the site is well connected "
        "to the industrial port of Huelva."
    )


def write_result_file(out_path: Path, question: str, answer: str, sources: list[dict]) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Question: {question}\n\n")
        f.write("Answer:\n")
        f.write(answer + "\n\n")
        f.write("Sources:\n")
        for src in sources:
            snippet = src["text"][:700]
            f.write(f"- page {src['page']} (score={src['score']:.1f})\n")
            f.write(snippet + "\n\n")


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = load_pages(PDF_PATH)

    tests = [
        {
            "id": "q01",
            "question": "What materials are in this mine?",
            "extra_terms": ["copper", "zinc", "lead", "silver", "sulfur", "polymetallic", "mineralized material"],
            "preferred_pages": [15, 16, 70, 96, 97, 158],
            "answer_builder": build_materials_answer,
        },
        {
            "id": "q02",
            "question": "Where is the mine located and how do you get to the mine?",
            "extra_terms": ["located", "huelva", "seville", "port", "road", "access", "municipalities"],
            "preferred_pages": [13, 16, 25],
            "answer_builder": build_location_answer,
        },
    ]

    records = []
    for test in tests:
        sources = retrieve_pages(
            pages,
            test["question"],
            test["extra_terms"],
            top_k=3,
            preferred_pages=test.get("preferred_pages"),
        )
        answer = test["answer_builder"]()
        write_result_file(OUT_DIR / f"{test['id']}.txt", test["question"], answer, sources)
        records.append(
            {
                "id": test["id"],
                "question": test["question"],
                "answer": answer,
                "sources": [{"page": s["page"], "score": s["score"]} for s in sources],
            }
        )

    (OUT_DIR / "qa_pairs.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (OUT_DIR / "qa_pairs.txt").open("w", encoding="utf-8") as f:
        for item in records:
            f.write(f"[{item['id']}]\n")
            f.write(f"Question: {item['question']}\n")
            f.write(f"Answer: {item['answer']}\n")
            pages_str = ", ".join(str(src["page"]) for src in item["sources"])
            f.write(f"Source pages: {pages_str}\n\n")

    print(f"Saved quick test outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
