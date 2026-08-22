"""Groq-backed comparison answers, with a deterministic offline fallback.

Mirrors closing-salesman-mvp/src/closing_salesman/answer.py's pattern: any
Groq failure (missing key, rate limit, bad JSON) degrades to a data-only
answer computed straight from the product schema — the user never sees a
hard error, they just see an "offline" badge instead of "AI".
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config import GROQ_CHAT_MODEL, get_secret

logger = logging.getLogger(__name__)

COMPARE_SYSTEM_PROMPT = """You are a shopping assistant helping a Myntra user compare 2-3 wishlist items and decide which one to buy.

You are given structured product JSON for each item (price, material, fit, rating, reviews, return policy, trust signals) and the user's comparison question or follow-up.

Rules:
1. Base every factual claim primarily on the provided product JSON.
2. Recommend exactly ONE product as the best fit for what the user asked, and put its sku_id in "winner_sku".
3. Return ONLY a JSON object with exactly these keys:
   {"answer": "<2-4 plain, conversational sentences, no markdown>", "winner_sku": "<sku_id of the recommended product>"}
4. Never invent numbers (prices, ratings, review counts) that are not in the data.
5. If this is a follow-up question, use the prior conversation for context but keep the same JSON output shape.
"""


def _slim_product(p: dict) -> dict:
    """Send Groq only the fields relevant to a comparison decision — not the
    full catalog entry (drops the raw sample_reviews list, keeps the
    pre-summarized trust_signals top_praise/top_complaint instead)."""
    return {
        "sku_id": p["sku_id"],
        "title": p["title"],
        "brand": p["brand"],
        "price": p["price"],
        "material": p.get("material", {}),
        "fit": p.get("fit", {}),
        "rating": p.get("rating"),
        "review_count": p.get("review_count"),
        "return_policy": p.get("return_policy", {}),
        "trust_signals": p.get("trust_signals", {}),
    }


def call_groq_compare(products: list[dict], history: list[dict]) -> dict[str, Any] | None:
    """Returns {"answer", "winner_sku", "groq_called": True} on success,
    or None if Groq is unavailable/failed (caller should use fallback_answer)."""
    key = get_secret("GROQ_API_KEY")
    if not key:
        logger.info("Myntra Compare UI: GROQ_API_KEY not set, using offline fallback")
        return None

    from groq import RateLimitError, Groq

    model = get_secret("GROQ_CHAT_MODEL", GROQ_CHAT_MODEL)
    client = Groq(api_key=key, max_retries=3)

    product_context = json.dumps([_slim_product(p) for p in products], indent=2)
    messages = [{"role": "system", "content": COMPARE_SYSTEM_PROMPT + "\n\nProducts being compared:\n" + product_context}]
    for m in history:
        role = "assistant" if m.get("role") == "assistant" else "user"
        messages.append({"role": role, "content": m.get("text", "")})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
    except RateLimitError as exc:
        logger.warning("Myntra Compare UI: Groq rate limited: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 — any Groq/API failure degrades to fallback
        logger.warning("Myntra Compare UI: Groq call failed: %s", exc)
        return None

    content = (response.choices[0].message.content or "{}").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning("Myntra Compare UI: bad JSON from Groq: %s", exc)
        return None

    if not isinstance(parsed, dict) or "answer" not in parsed or "winner_sku" not in parsed:
        logger.warning("Myntra Compare UI: Groq response missing required keys: %s", parsed)
        return None

    valid_skus = {p["sku_id"] for p in products}
    if parsed["winner_sku"] not in valid_skus:
        parsed["winner_sku"] = products[0]["sku_id"]

    parsed["groq_called"] = True
    return parsed


# ---------------------------------------------------------------------------
# Offline fallback — a Python port of the same keyword-focus heuristic used
# client-side in static/index.html before Groq was wired in, kept here so a
# missing/rate-limited key still produces a sensible, explainable answer.
# ---------------------------------------------------------------------------

_FOCUS_KEYWORDS = {
    "price": ["economical", "cheap", "budget", "affordable", "least expensive", "value for money", "cheapest", "pocket friendly", "inexpensive"],
    "premium": ["best looking", "premium", "luxury", "stylish", "looks", "elegant", "expensive", "classy", "good looking"],
    "fit": ["office", "formal", "summer", "comfortable", "comfort", "fit", "breathable", "casual", "work", "daily wear"],
    "trust": ["trust", "reliable", "genuine", "authentic", "review", "reviews", "fake"],
    "returns": ["return", "exchange", "refund", "risk", "send back"],
}


def _classify_focus(text: str) -> str:
    t = (text or "").lower()
    best, best_score = "general", 0
    for focus, kws in _FOCUS_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in t)
        if score > best_score:
            best, best_score = focus, score
    return best


def _is_premium(p: dict) -> bool:
    s = (p["material"].get("composition", "") + " " + p["material"].get("feel_notes", "")).lower()
    return any(k in s for k in ("genuine leather", "premium", "soft grain", "structured"))


def _is_breathable(p: dict) -> bool:
    s = p["material"].get("feel_notes", "").lower()
    return "breathable" in s or "mesh" in s


def _return_score(p: dict) -> int:
    rp = p.get("return_policy", {})
    score = rp.get("window_days", 0)
    if rp.get("type") == "return-and-refund":
        score += 15
    if rp.get("who_pays_return_shipping") == "brand":
        score += 10
    return score


def fallback_answer(products: list[dict], query: str) -> dict[str, Any]:
    focus = _classify_focus(query)

    if focus == "price":
        winner = min(products, key=lambda p: p["price"])
        reason = f"it's the most affordable of the {len(products)} at ₹{winner['price']:,}"
    elif focus == "premium":
        premium_ones = [p for p in products if _is_premium(p)]
        pool = premium_ones or products
        winner = max(pool, key=(lambda p: p["price"]) if premium_ones else (lambda p: p["rating"]))
        reason = f"its material — \"{winner['material'].get('feel_notes', '')}\" — reads more premium than the others"
    elif focus == "fit":
        true_to_size = [p for p in products if "true to size" in p["fit"].get("review_consensus", "").lower()]
        breathable = [p for p in products if _is_breathable(p)]
        winner = next((p for p in true_to_size if p in breathable), None) or (true_to_size[0] if true_to_size else None) \
            or (breathable[0] if breathable else None) or max(products, key=lambda p: p["rating"])
        reason = f"its fit consensus — \"{winner['fit'].get('review_consensus', '')}\""
        if _is_breathable(winner):
            reason += ", and it runs breathable, good for daily/summer wear"
    elif focus == "trust":
        winner = max(products, key=lambda p: p["trust_signals"].get("verified_purchase_pct", 0))
        reason = f"{winner['trust_signals'].get('verified_purchase_pct', 0)}% of its reviews are verified purchases, the highest here"
    elif focus == "returns":
        winner = max(products, key=_return_score)
        rp = winner["return_policy"]
        reason = f"it has the safest return policy — {rp.get('window_days', '?')}-day {rp.get('type', 'policy').replace('-', ' ')}, {rp.get('who_pays_return_shipping', 'you')} pays shipping"
    else:
        winner = max(products, key=lambda p: p["rating"])
        reason = f"it has the strongest overall rating ({winner['rating']}★ across {winner['review_count']} reviews)"

    focus_label = {
        "price": "most economical", "premium": "best looking", "fit": "best suited for what you described",
        "trust": "most trustworthy", "returns": "lowest-risk to buy", "general": "overall best",
    }[focus]

    answer = f"Based on \"{query or 'your wishlist'}\", I'd go with the {winner['brand']} {' '.join(winner['title'].split()[-2:])} — it's the {focus_label} pick: {reason}."
    return {"answer": answer, "winner_sku": winner["sku_id"], "groq_called": False}
