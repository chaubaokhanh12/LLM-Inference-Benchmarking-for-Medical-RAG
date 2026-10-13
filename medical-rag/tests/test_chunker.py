import unicodedata

import pytest

from medrag.ingest.chunker import Chunker, ChunkerConfig, classify_lines, split_sentences
from medrag.ingest.cleaner import clean_text
from medrag.types import Document


@pytest.fixture(scope="module")
def chunker() -> Chunker:
    return Chunker(ChunkerConfig())


def make_doc(text: str, title: str = "Tài liệu thử", lang: str = "vi") -> Document:
    return Document(doc_id="test/doc", title=title, lang=lang, text=clean_text(text),
                    source_path="test/doc.md")


def filler(n: int, tag: str = "") -> str:
    return " ".join(
        f"Bệnh nhân {tag}cần được theo dõi chặt chẽ các dấu hiệu sinh tồn trong lần khám thứ {i}."
        for i in range(n)
    )


DOSE_TABLE = "\n".join(
    ["| Cân nặng | Liều paracetamol | Số lần mỗi ngày |", "| --- | --- | --- |"]
    + [f"| {5 + 3 * i} kg | {75 + 45 * i} mg | 4 lần |" for i in range(10)]
)


# ---------------------------------------------------------------- required by the spec


def test_dose_table_stays_in_one_chunk(chunker):
    text = f"## Liều dùng\n\n{filler(12)}\n\nBảng 1. Liều theo cân nặng\n{DOSE_TABLE}\n\n{filler(12)}"
    chunks = chunker.chunk(make_doc(text))
    holders = [c for c in chunks if "| Cân nặng |" in c.text]
    assert len(holders) == 1
    table_chunk = holders[0]
    for i in range(10):
        assert f"| {5 + 3 * i} kg |" in table_chunk.text
    assert "Bảng 1. Liều theo cân nặng" in table_chunk.text  # caption kept with its table
    assert sum(c.text.count("| --- | --- | --- |") for c in chunks) == 1  # never duplicated


def test_chuong_dieu_section_path(chunker):
    text = (
        "Chương I. Quy định chung\n\nĐiều 1. Phạm vi điều chỉnh\n\n" + filler(3, "A ")
        + "\n\nĐiều 2. Đối tượng áp dụng\n\n" + filler(3, "B ")
        + "\n\nChương II. Chẩn đoán\n\nĐiều 3. Tiêu chuẩn\n\n" + filler(3, "C ")
    )
    paths = {tuple(c.section_path): c.text for c in chunker.chunk(make_doc(text))}
    assert ("Chương I. Quy định chung", "Điều 1. Phạm vi điều chỉnh") in paths
    assert ("Chương I. Quy định chung", "Điều 2. Đối tượng áp dụng") in paths
    assert ("Chương II. Chẩn đoán", "Điều 3. Tiêu chuẩn") in paths
    assert "B cần" in paths[("Chương I. Quy định chung", "Điều 2. Đối tượng áp dụng")]


def test_no_chunk_exceeds_512_tokens_unless_oversized(chunker):
    big_table = "\n".join(
        ["| Thuốc | Liều | Ghi chú |", "| --- | --- | --- |"]
        + [f"| Thuốc số {i} | {i * 10} mg mỗi ngày chia hai lần | Thận trọng khi suy thận |"
           for i in range(80)]
    )
    text = (f"# Tổng quan\n\n{filler(60)}\n\n## Điều trị\n\n{filler(40)}\n\n{big_table}\n\n"
            f"{filler(30)}\n\n- {filler(2)}\n- {filler(2)}")
    chunks = chunker.chunk(make_doc(text))
    assert any(c.oversized for c in chunks)
    tok = chunker.counter
    for c in chunks:
        real = tok.count_with_specials(f"passage: {c.header}\n{c.text}")
        assert real == c.n_tokens
        if not c.oversized:
            assert c.n_tokens <= 512, c.chunk_id


# ---------------------------------------------------------------- extra edge cases


def test_numbered_list_is_not_heading():
    lines = classify_lines("Triệu chứng gồm:\n1. Sốt cao\n2. Đau đầu\n3. Buồn nôn", ChunkerConfig())
    assert [ln.kind for ln in lines] == ["text", "list", "list", "list"]


def test_numbered_heading_followed_by_paragraph():
    lines = classify_lines("2. Điều trị\nBệnh nhân cần nghỉ ngơi.", ChunkerConfig())
    assert lines[0].kind == "heading"


def test_species_abbreviation_is_not_roman_heading():
    lines = classify_lines("V. cholerae gây bệnh tả.\nI. ĐẠI CƯƠNG", ChunkerConfig())
    assert lines[0].kind == "text"
    assert lines[1].kind == "heading"


def test_mixed_heading_hierarchy(chunker):
    text = (
        "I. ĐẠI CƯƠNG\n\n1. Định nghĩa\n\n" + filler(2, "a ")
        + "\n\n1.1. Dịch tễ\n\n" + filler(2, "b ")
        + "\n\n2. Nguyên nhân\n\n" + filler(2, "c ")
        + "\n\nII. ĐIỀU TRỊ\n\n" + filler(2, "d ")
    )
    paths = [tuple(c.section_path) for c in chunker.chunk(make_doc(text))]
    assert paths == [
        ("I. ĐẠI CƯƠNG", "1. Định nghĩa"),
        ("I. ĐẠI CƯƠNG", "1. Định nghĩa", "1.1. Dịch tễ"),
        ("I. ĐẠI CƯƠNG", "2. Nguyên nhân"),
        ("II. ĐIỀU TRỊ",),
    ]


def test_unaccented_structure_headings():
    lines = classify_lines("Chuong I\nDieu 1. Pham vi\nMục tiêu điều trị là hạ sốt.", ChunkerConfig())
    assert [ln.kind for ln in lines] == ["heading", "heading", "text"]


def test_list_kept_with_its_intro(chunker):
    items = "\n".join(f"- Triệu chứng số {i}: sốt, đau đầu, đau cơ, phát ban" for i in range(15))
    text = f"## Lâm sàng\n\n{filler(25)}\n\nCác triệu chứng thường gặp:\n{items}\n\n{filler(5)}"
    chunks = chunker.chunk(make_doc(text))
    holders = [c for c in chunks if "Triệu chứng số 0" in c.text]
    assert len(holders) == 1
    assert "Các triệu chứng thường gặp:" in holders[0].text
    assert "Triệu chứng số 14" in holders[0].text


def test_no_overlap_across_sections(chunker):
    text = f"## Phần A\n\n{filler(40, 'A ')}\n\n## Phần B\n\n{filler(40, 'B ')}"
    chunks = chunker.chunk(make_doc(text))
    for c in chunks:
        if c.section_path == ["Phần B"]:
            assert "A cần" not in c.text
    a_chunks = [c for c in chunks if c.section_path == ["Phần A"]]
    assert len(a_chunks) >= 2
    # consecutive chunks inside one section share one sentence
    last_sentence = split_sentences(a_chunks[0].text)[-1]
    assert a_chunks[1].text.startswith(last_sentence)


def test_nfd_input_gives_same_chunks(chunker):
    text = f"## Điều trị\n\n{filler(10)}"
    nfd = unicodedata.normalize("NFD", text)
    a = chunker.chunk(make_doc(text))
    b = chunker.chunk(make_doc(nfd))
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_chunk_ids_are_deterministic(chunker):
    text = f"## A\n\n{filler(30)}\n\n## B\n\n{filler(30)}"
    ids = [c.chunk_id for c in chunker.chunk(make_doc(text))]
    assert ids == [f"test/doc:{i}" for i in range(len(ids))]
    assert ids == [c.chunk_id for c in chunker.chunk(make_doc(text))]


def test_tiny_parent_section_merged_into_child(chunker):
    text = f"## Điều trị\n\nGồm hai nhóm:\n\n### Thuốc\n\n{filler(5)}"
    chunks = chunker.chunk(make_doc(text))
    assert chunks[0].section_path == ["Điều trị", "Thuốc"]
    assert chunks[0].text.startswith("Gồm hai nhóm:")


def test_split_oversized_repeats_table_header():
    rows = [f"| Thuốc {i} | {i} mg | Thận trọng khi suy gan và suy thận nặng |" for i in range(90)]
    table = "\n".join(["| Thuốc | Liều | Ghi chú |", "| --- | --- | --- |", *rows])
    chunker = Chunker(ChunkerConfig(split_oversized=True))
    chunks = chunker.chunk(make_doc(f"## Bảng\n\n{table}"))
    assert len(chunks) >= 2
    assert all(c.text.startswith("| Thuốc | Liều | Ghi chú |") for c in chunks)
    assert all(not c.oversized and c.n_tokens <= 512 for c in chunks)
    assert sum(c.text.count("| Thuốc 1") for c in chunks) >= 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sốt cao. Đau đầu dữ dội. Uống nhiều nước.", 3),
        ("Dùng 0.5 mg/kg. Trẻ em dưới 2 tuổi cần thận trọng.", 2),
        ("Do vi khuẩn E. coli gây ra. Điều trị bằng kháng sinh.", 2),
        ("Theo Dr. Smith, bệnh lành tính. Tuy vậy cần theo dõi.", 2),
        ("Gồm sốt, ho, v.v. Các triệu chứng khác hiếm gặp.", 2),
    ],
)
def test_split_sentences(text, expected):
    assert len(split_sentences(text)) == expected
