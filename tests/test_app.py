from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_single_page_contract_without_exception() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert app.selectbox(key="parse_document").value == "bluebird_supplier_native"
    assert any(caption.value == "Page 1 of 1" for caption in app.caption)


def test_app_renders_page_slider_for_multi_page_contract() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.selectbox(key="parse_document").set_value("northstar_msa_scan").run()

    assert not app.exception
    assert app.slider(key="parse_page").value == 1
