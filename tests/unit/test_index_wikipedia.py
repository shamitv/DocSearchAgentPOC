import pytest
import mwparserfromhell
from index_wikipedia import extract_plain_text, _process_templates


def test_plain_text_simple():
    assert extract_plain_text("Hello world") == "Hello world"


def test_strip_markup():
    assert extract_plain_text("'''bold''' and ''italic''") == "bold and italic"


def test_template_simple():
    raw = "{{Test|a=1|b=2}} end"
    # Expect actual newline
    expected = "a: 1\nb: 2 end"
    assert extract_plain_text(raw) == expected


def test_nested_templates():
    raw = "{{Outer|x={{Inner|y=z}}|w=4}}"
    # Expect actual newline, and correct nested processing
    # The current function processes outer first, leaving inner unprocessed
    # Let's adjust the test to reflect the *desired* behavior (inner processed first)
    # Although the function might still fail this until fixed
    expected = "x: y: z\nw: 4"
    assert extract_plain_text(raw) == expected


def test_process_templates_simple():
    raw = "{{Test|a=1|b=2}} end"
    wikicode = mwparserfromhell.parse(raw)
    _process_templates(wikicode)
    # Expect actual newline
    expected = "a: 1\nb: 2 end"
    assert str(wikicode) == expected

def test_process_templates_nested():
    # Note: The current implementation replaces the inner template first, then the outer.
    # The replacement text becomes part of the outer template's parameters if not handled carefully.
    # This test reflects the *current* behavior.
    raw = "{{Outer|x={{Inner|y=z}}|w=4}}"
    wikicode = mwparserfromhell.parse(raw)
    _process_templates(wikicode)
    # The inner template {{Inner|y=z}} is replaced by "y: z"
    # The outer template becomes {{Outer|x=y: z|w=4}}
    # This is then replaced by "x: y: z\\nw: 4"
    expected = "x: y: z\nw: 4"
    assert str(wikicode) == expected

def test_process_templates_no_templates():
    raw = "Just plain text."
    wikicode = mwparserfromhell.parse(raw)
    _process_templates(wikicode)
    assert str(wikicode) == raw

def test_process_templates_multiple():
    raw = "Start {{T1|p=1}} middle {{T2|q=2}} end"
    wikicode = mwparserfromhell.parse(raw)
    _process_templates(wikicode)
    expected = "Start p: 1 middle q: 2 end"
    assert str(wikicode) == expected

def test_process_templates_params_with_spaces():
    raw = "{{Spaced | key = value | another key = another value }}"
    wikicode = mwparserfromhell.parse(raw)
    _process_templates(wikicode)
    # Expect actual newline
    expected = "key: value\nanother key: another value"
    assert str(wikicode) == expected

def test_extract_postnominals():
    raw = "{{postnominals|country=GBR|size=100%|OBE|FRSL}}"
    # Parameters: country=GBR, size=100%, 1=OBE, 2=FRSL
    # Expect actual newlines
    expected = "country: GBR\nsize: 100%\n1: OBE\n2: FRSL"
    assert extract_plain_text(raw) == expected

def test_extract_pipe_template():
    raw = "{{!}}"
    # This template usually renders as '|', but our function extracts parameters.
    # Since it has no parameters, the replacement is empty.
    expected = ""
    assert extract_plain_text(raw) == expected

def test_extract_langx():
    raw = "{{langx|el|Αθήνα|Athína}}"
    # Parameters: 1=el, 2=Αθήνα, 3=Athína
    # Expect actual newlines
    expected = "1: el\n2: Αθήνα\n3: Athína"
    assert extract_plain_text(raw) == expected

def test_extract_native_name():
    raw = "{{native name|el|Αθήνα}}"
    # Parameters: 1=el, 2=Αθήνα
    # Expect actual newlines
    expected = "1: el\n2: Αθήνα"
    assert extract_plain_text(raw) == expected

def test_extract_notetag():
    raw = "{{NoteTag|With various theological and doctrinal identities, including Anglo-Catholic, Liberal, Evangelical}}"
    # Parameters: 1=With various...
    expected = "1: With various theological and doctrinal identities, including Anglo-Catholic, Liberal, Evangelical"
    assert extract_plain_text(raw) == expected

def test_extract_collapsible_list():
    raw = "{{collapsible list|'''Non-Pereil Publishing Group''' (#1–67)<br>'''Perfect Film & Chemical Corp.''' (#68–69)<br>'''Magazine Management Co.''' (#70–118)<br>'''[[Marvel Comics]]''' (#119–present)}}"
    # Parameter 1 contains the list with markup.
    # _process_templates replaces the template with "1: [content]"
    # strip_code removes bold ('''), wikilinks ([[]]), and converts <br> to newline.
    expected = "1: Non-Pereil Publishing Group (#1–67)\nPerfect Film & Chemical Corp. (#68–69)\nMagazine Management Co. (#70–118)\nMarvel Comics (#119–present)"
    assert extract_plain_text(raw) == expected
