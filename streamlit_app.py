"""Myntra Compare UI — standalone entry point.

Second-UI companion to the closing-salesman-mvp / myntra-review-discovery-engine
repos: a wishlist screen with a "Want help in comparing?" flow (select up to 3
items, state what you want to compare on, get a Groq-backed recommendation +
side-by-side table, or a regret screen when items are too different).

Frontend is a CDN-based React + Tailwind app (no Node/build step) in
static/index.html, wired to this Python backend via Streamlit's official
static-component postMessage protocol (components.declare_component) so the
Groq call happens server-side — the API key never reaches the browser.
Secrets (GROQ_API_KEY / GROQ_CHAT_MODEL) are read from .streamlit/secrets.toml,
same as the sibling repos, so this is ready to deploy to Streamlit Community
Cloud as-is (paste .streamlit/secrets.toml.template into the app's Secrets).
"""

from __future__ import annotations

import json
import logging

import streamlit as st
import streamlit.components.v1 as components

from config import PRODUCTS_FILE, STATIC_DIR
from groq_answer import call_groq_compare, fallback_answer

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Want help in comparing? — Myntra",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Strip Streamlit's own chrome (header, block-container padding) and stretch
# the component iframe to cover the full browser viewport.
st.markdown(
    """
    <style>
        header[data-testid="stHeader"] { display: none; }
        .block-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
        html, body, .stApp { overflow: hidden !important; }
        iframe { position: fixed; inset: 0; width: 100vw !important; height: 100vh !important; border: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

_compare_component = components.declare_component("myntra_compare", path=str(STATIC_DIR))


@st.cache_data
def _load_catalog() -> tuple[list[dict], dict[str, dict]]:
    raw = json.loads(PRODUCTS_FILE.read_text(encoding="utf-8"))
    clusters = raw["clusters"]
    sku_lookup: dict[str, dict] = {}
    for cluster in clusters:
        for product in cluster["products"]:
            sku_lookup[product["sku_id"]] = {**product, "category": cluster["category"], "cluster_id": cluster["cluster_id"]}
    return clusters, sku_lookup


def main() -> None:
    clusters, sku_lookup = _load_catalog()

    if "pending_answers" not in st.session_state:
        st.session_state.pending_answers = {}
    if "processed_action_ids" not in st.session_state:
        st.session_state.processed_action_ids = set()

    value = _compare_component(
        wishlist=clusters,
        pendingAnswers=st.session_state.pending_answers,
        key="myntra_compare",
        default=None,
    )

    if isinstance(value, dict):
        action_id = value.get("actionId")
        action = value.get("action")
        if action_id and action_id not in st.session_state.processed_action_ids and action in ("submit_query", "follow_up"):
            st.session_state.processed_action_ids.add(action_id)

            selected_skus = value.get("selectedSkus", [])
            products = [sku_lookup[sku] for sku in selected_skus if sku in sku_lookup]
            history = value.get("messages", [])
            latest_query = history[-1]["text"] if history else ""

            if products:
                result = call_groq_compare(products, history)
                if result is None:
                    result = fallback_answer(products, latest_query)
            else:
                result = {"answer": "Something went wrong — no valid products were selected.", "winner_sku": None, "groq_called": False}

            st.session_state.pending_answers[action_id] = result
            st.rerun()


if __name__ == "__main__":
    main()
