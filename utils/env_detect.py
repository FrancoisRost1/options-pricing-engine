"""
Environment detection -- identifies Streamlit Cloud vs local.

Streamlit Cloud runs from /mount/src/ and sets STREAMLIT_SHARING_MODE.
When on Cloud, yfinance is blocked (Yahoo rate-limits Cloud IPs),
so the app must default to synthetic data.
"""

import os


def is_streamlit_cloud() -> bool:
    """
    Detect whether the app is running on Streamlit Cloud.

    Checks two signals:
      1. /mount/src/ directory exists (Cloud container layout)
      2. STREAMLIT_SHARING_MODE env var is set
    Either is sufficient.
    """
    if os.path.isdir("/mount/src"):
        return True
    if os.environ.get("STREAMLIT_SHARING_MODE"):
        return True
    return False
