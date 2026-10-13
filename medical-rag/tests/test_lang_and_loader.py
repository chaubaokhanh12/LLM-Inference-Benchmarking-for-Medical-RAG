import pytest

from medrag.ingest.cleaner import clean_text, remove_page_furniture
from medrag.ingest.loader import load_document, split_front_matter
from medrag.lang import detect_lang
from medrag.text import slugify, strip_accents


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Sốt xuất huyết có nguy hiểm không?", "vi"),
        ("What is the dose of paracetamol for children?", "en"),
        ("dau dau uong thuoc gi", "vi"),
        ("benh tieu duong co chua khoi khong", "vi"),
        ("Metformin side effects", "en"),
        ("Can I take ibuprofen with aspirin?", "en"),
        ("", "en"),
    ],
)
def test_detect_lang(text, lang):
    assert detect_lang(text) == lang


def test_slugify_and_strip_accents():
    assert strip_accents("Đau đầu") == "Dau dau"
    assert slugify("Sốt xuất huyết Dengue") == "sot_xuat_huyet_dengue"
    assert slugify("HIV/AIDS") == "hiv_aids"


def test_clean_text_keeps_columns_and_removes_invisible():
    raw = "Thuốc   Liều\r\nA\u200b      5 mg  \n\n\n\nB\u00a0x"
    out = clean_text(raw)
    assert out == "Thuốc   Liều\nA      5 mg\n\nB x"


def test_remove_page_furniture():
    pages = [f"BỘ Y TẾ\nNội dung trang {i}\n{i + 1}" for i in range(4)]
    out = remove_page_furniture(pages)
    assert all("BỘ Y TẾ" not in p for p in out)
    assert out[0] == "Nội dung trang 0"


def test_front_matter_and_loader(tmp_path):
    root = tmp_path / "raw"
    (root / "wiki" / "vi").mkdir(parents=True)
    f = root / "wiki" / "vi" / "Sốt xuất huyết.md"
    f.write_text("---\ntitle: Sốt xuất huyết Dengue\nlang: vi\nurl: https://x.y/z\n---\n\n## A\nB",
                 encoding="utf-8")
    doc = load_document(f, root)
    assert doc.doc_id == "wiki/vi/sot_xuat_huyet"
    assert doc.title == "Sốt xuất huyết Dengue"
    assert doc.lang == "vi"
    assert doc.meta["url"] == "https://x.y/z"
    assert doc.text == "## A\nB"
    assert split_front_matter("no front matter") == ({}, "no front matter")
