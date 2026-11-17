"""Headless Streamlit UI test (no browser). Run from the project root:  python tests/run_ui_test.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
assert not at.exception, at.exception
print("Dashboard loads")
at.button[0].click().run()   # quick demo button
assert not at.exception, at.exception
print("Quick demo ran; history rows:", len(at.session_state.history))
for page in ["🖼️ Image Detection", "🎞️ Video Detection", "🎯 Object Tracking",
             "📊 Analytics & Evaluation", "ℹ️ About Project"]:
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, (page, at.exception)
    print("OK:", page)
print("ALL UI PAGES RENDER WITHOUT ERRORS")
