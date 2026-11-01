from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import ButtonGroup


def test_ccd_selection_and_demo_survive_tab_switch(monkeypatch):
    # Streamlit 1.52's AppTest treats segmented-control options as strings
    # although they are protobuf objects, and iterates single selections.
    def indices(widget):
        value = widget.value
        values = [value] if isinstance(value, str) else value
        return [next(i for i, option in enumerate(widget.options)
                     if option == widget.format_func(item)) for item in values]

    monkeypatch.setattr(ButtonGroup, "indices", property(indices))
    app = AppTest.from_file("app.py", default_timeout=30).run()
    assert not app.exception
    app.radio(key="design_selection_widget").set_value("中心複合計画（CCD）").run()
    assert not app.exception
    app.get("button_group")[0].set_value(["データ"]).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 15
    app.get("button_group")[0].set_value(["実験計画"]).run()
    assert not app.exception
    assert app.radio(key="design_selection_widget").value == "中心複合計画（CCD）"
