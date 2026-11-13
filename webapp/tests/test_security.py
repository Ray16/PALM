"""Webapp security settings (moved from PALM/tests/test_improvements.py).

Run from the directory containing PALM/:  python -m pytest PALM/webapp/tests
"""
from pathlib import Path


def test_path_traversal_sanitized():
    safe = Path("../../../etc/passwd").name
    assert safe == "passwd"


def test_security_limits_configured():
    from PALM.webapp.app import _allowed_origins, MAX_UPLOAD_BYTES, RATE_LIMIT_MAX
    assert _allowed_origins is not None
    assert MAX_UPLOAD_BYTES > 0
    assert RATE_LIMIT_MAX > 0
