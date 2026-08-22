"""App configuration — mirrors closing-salesman-mvp/src/config.py's get_secret()
pattern (st.secrets first, then os.getenv) so this app reads the same
GROQ_API_KEY / GROQ_CHAT_MODEL secrets whether run locally or on Streamlit
Community Cloud, sourced from .streamlit/secrets.toml.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"
PRODUCTS_FILE = DATA_DIR / "sample_products.json"


def get_secret(key: str, default: str = "") -> str:
    """Retrieve secret/config value prioritizing Streamlit st.secrets over os.getenv."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


GROQ_CHAT_MODEL = get_secret("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
