import pytest
from index_wikipedia import extract_plain_text


def test_plain_text_simple():
    assert extract_plain_text("Hello world") == "Hello world"


def test_strip_markup():
    assert extract_plain_text("'''bold''' and ''italic''") == "bold and italic"


def test_template_simple():
    raw = "{{Test|a=1|b=2}} end"
    expected = "a: 1\nb: 2 end"
    assert extract_plain_text(raw) == expected


def test_nested_templates():
    raw = "{{Outer|x={{Inner|y=z}}|w=4}}"
    expected = "x: y: z\nw: 4"
    assert extract_plain_text(raw) == expected
