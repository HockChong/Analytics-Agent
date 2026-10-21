from eval.score_tables import score_cell

GOLD = {"row": "Net Property Income", "column": "2025", "value": "1,189.7", "bbox": [1109.3, 253.1, 1143.8, 262.4]}


def cell(**kw):
    return {"row": "Net Property Income", "column": "CICT Group | 2025", "value": "1,189.7",
            "bbox": [1109.5, 253.0, 1143.7, 262.5], **kw}


def test_correct_when_row_column_and_value_match():
    assert score_cell(GOLD, [cell()]) == "correct"


def test_note_reference_and_case_are_ignored_in_row_labels():
    assert score_cell(GOLD, [cell(row="net property income (Note 21)")]) == "correct"


def test_wrong_year_is_wrong_column():
    assert score_cell(GOLD, [cell(column="2024")]) == "wrong_column"


def test_wrong_row_and_wrong_value():
    assert score_cell(GOLD, [cell(row="Gross Revenue")]) == "wrong_row"
    assert score_cell(GOLD, [cell(value="1,189.1")]) == "wrong_value"


def test_cell_elsewhere_on_the_page_is_missing():
    assert score_cell(GOLD, [cell(bbox=[1109.3, 300.0, 1143.8, 309.0])]) == "missing"
    assert score_cell(GOLD, []) == "missing"
