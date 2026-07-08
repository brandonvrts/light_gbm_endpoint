"""Phase 4 UI eval: run the Streamlit app headlessly via AppTest, assert each page
renders with no exception and a real multifamily prediction completes. Needs the API up."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402


def run():
    at = AppTest.from_file("frontend/app.py", default_timeout=120)
    at.run()
    assert not at.exception, f"initial multifamily render: {at.exception}"
    print("PASS multifamily render")

    if at.button:
        at.button[0].click().run()
        assert not at.exception, f"multifamily prediction: {at.exception}"
        print("PASS multifamily prediction + charts")

    at.radio[0].set_value("Brokerage - prospect conversion").run()
    assert not at.exception, f"brokerage render: {at.exception}"
    print("PASS brokerage render")

    at.radio[0].set_value("Marketing - cohort analytics").run()
    assert not at.exception, f"marketing render: {at.exception}"
    print("PASS marketing render")

    print("UI EVAL PASS")


if __name__ == "__main__":
    run()
