"""Every page runs headlessly through Streamlit's own script runner and draws something.

A page that raises on load is invisible to every other test here; so is a page that
runs clean and renders nothing."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
VIEWS = sorted((APP / "views").glob("*.py"))


@pytest.mark.parametrize("view", VIEWS, ids=lambda path: path.stem)
def test_the_page_runs_without_raising_and_renders(view):
    app = AppTest.from_file(str(view), default_timeout=240).run()
    assert not app.exception, [exception.message for exception in app.exception]
    assert len(app.title) == 1
    assert len(app.markdown) + len(app.metric) + len(app.dataframe) >= 3


def test_the_entry_point_opens_on_the_forecast():
    app = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=240).run()
    assert not app.exception, [exception.message for exception in app.exception]
    assert app.title[0].value.startswith("Where new fires")
