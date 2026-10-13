# Medical RAG — Đặc tả để dựng lại

Hệ thống hỏi đáp tài liệu y khoa song ngữ Việt–Anh, tự xây retrieval từ đầu (không dùng LangChain/LlamaIndex), phục vụ qua FastAPI, backend LLM thay đổi được qua `.env`.

Bản dựng lại này có thêm một mục tiêu: **sẵn sàng cho project LLM Inference Benchmarking**. Nghĩa là xuất được prompt cuối ra JSONL, ghi được thời gian từng giai đoạn, và cắm được vLLM làm backend.

---

## 1. Mục tiêu và phạm vi

**Phải có**

- Nạp tài liệu y khoa (Markdown, TXT, PDF đã chuyển sang text), tiếng Việt và tiếng Anh.
- Chunker hiểu cấu trúc văn bản, không cắt ngang bảng liều dùng hay danh sách triệu chứng.
- Hybrid retrieval: dense (bi-encoder) + BM25, gộp bằng Reciprocal Rank Fusion (RRF), sau đó rerank bằng cross-encoder.
- Sinh câu trả lời bằng LLM, có trích dẫn nguồn (chunk id, tên tài liệu, đề mục).
- REST API bằng FastAPI.
- Backend LLM cắm được: Ollama, OpenAI, Gemini, Groq, **vLLM** (qua API tương thích OpenAI).
- Đánh giá retrieval và câu trả lời bằng bộ QA tổng hợp (synthetic QA, kiểu Ragas).
- Xuất prompt cuối ra JSONL và đo thời gian từng giai đoạn, phục vụ benchmark.

**Không làm (ở bản này)**

- Giao diện web (chỉ API; có thể thêm sau).
- Quản lý người dùng, xác thực.
- Cơ sở dữ liệu vector phân tán.

---

## 2. Kiến trúc tổng quan

```
             ┌──────────── Ingestion (offline) ────────────┐
documents ──►│ loader → cleaner → section-aware chunker     │──► chunks.jsonl
             │          → embedder (passage:) → FAISS index │──► dense.index
             │          → tokenizer VI/EN → BM25 index      │──► bm25.pkl
             └──────────────────────────────────────────────┘

             ┌──────────────── Query (online) ─────────────────────────────┐
question ───►│ detect lang → dense search (query:) ┐                        │
             │             → BM25 search ──────────┴► RRF fuse → top-k      │
             │             → cross-encoder rerank → top-n                   │
             │             → prompt builder → LLM backend → answer + cites  │
             └──────────────────────────────────────────────────────────────┘
```

Thời gian từng bước online (retrieve, rerank, LLM TTFT, LLM total) được ghi lại cho mỗi request.

---

## 3. Công nghệ đề xuất

| Thành phần | Lựa chọn | Ghi chú |
| --- | --- | --- |
| Ngôn ngữ | Python 3.11+ | |
| API | FastAPI + Uvicorn | Có endpoint streaming (SSE) |
| Dense embedding | `intfloat/multilingual-e5-base` | **Bắt buộc** thêm tiền tố `query: ` cho câu hỏi và `passage: ` cho chunk |
| Vector index | FAISS (`IndexFlatIP` trên vector đã chuẩn hóa) | Đủ cho vài chục nghìn chunk |
| BM25 | `rank_bm25` (BM25Okapi) | |
| Tách từ tiếng Việt | `pyvi` hoặc `underthesea` | BM25 tiếng Việt cần tách từ ghép ("đau_đầu"), nếu không điểm số kém |
| Reranker | Cross-encoder đa ngôn ngữ, ví dụ `BAAI/bge-reranker-v2-m3` | Kiểm tra bản mới nhất trên Hugging Face |
| LLM client | SDK `openai` (dùng chung cho OpenAI, vLLM, Groq, Ollama qua endpoint `/v1`) + SDK Gemini | Một interface, nhiều backend |
| Cấu hình | `pydantic-settings` đọc `.env` | |
| Đánh giá | Tự viết metric + tùy chọn thư viện `ragas` | |
| Test | `pytest` | |

---

## 4. Cấu trúc repo

```
medical-rag/
├── .env.example
├── pyproject.toml
├── data/
│   ├── raw/                 # tài liệu gốc
│   ├── processed/           # chunks.jsonl
│   └── index/               # dense.index, bm25.pkl, meta.json
├── src/medrag/
│   ├── config.py            # Settings đọc từ .env
│   ├── ingest/
│   │   ├── loader.py        # đọc md/txt/pdf → Document
│   │   ├── cleaner.py       # chuẩn hóa unicode, khoảng trắng
│   │   └── chunker.py       # section-aware chunker
│   ├── index/
│   │   ├── dense.py         # embed + FAISS
│   │   ├── sparse.py        # tokenizer VI/EN + BM25
│   │   └── build.py         # CLI dựng index
│   ├── retrieve/
│   │   ├── hybrid.py        # dense + BM25 + RRF
│   │   └── rerank.py        # cross-encoder
│   ├── llm/
│   │   ├── base.py          # interface LLMBackend
│   │   ├── openai_compat.py # OpenAI, vLLM, Groq, Ollama(/v1)
│   │   └── gemini.py
│   ├── generate/
│   │   └── prompt.py        # dựng prompt + định dạng trích dẫn
│   ├── pipeline.py          # ghép retrieve → rerank → generate, đo thời gian
│   ├── api/main.py          # FastAPI app
│   └── tools/
│       └── export_prompts.py # xuất prompt cuối ra JSONL cho benchmark
├── eval/
│   ├── make_synthetic_qa.py # sinh bộ QA từ chunk bằng LLM
│   ├── eval_retrieval.py    # Recall@k, MRR@k, nDCG@k
│   └── eval_answers.py      # faithfulness, answer relevance
└── tests/
```

---

## 5. Đặc tả từng module

### 5.1 Ingestion và chunker

**Kiểu dữ liệu**

```python
@dataclass
class Chunk:
    chunk_id: str          # f"{doc_id}:{index}"
    doc_id: str
    title: str             # tên tài liệu
    section_path: list[str]  # ví dụ ["Chương II", "Mục 3", "Điều 12"]
    text: str
    lang: str              # "vi" | "en"
    n_tokens: int
```

**Quy tắc của section-aware chunker**

1. Nhận diện ranh giới đề mục theo thứ tự ưu tiên:
   - Heading Markdown: `^#{1,6}\s`
   - Số La Mã: `^(I|II|III|IV|V|VI|VII|VIII|IX|X)+\.\s`
   - Cấu trúc tiếng Việt: `^(Chương|Mục|Điều|Phần)\s+[\dIVX]+` (không phân biệt hoa thường, chấp nhận cả bản không dấu `Chuong|Muc|Dieu`)
   - Đề mục đánh số: `^\d+(\.\d+)*\.?\s+[A-ZÀ-Ỹ]`
2. Tách văn bản thành các section theo ranh giới trên, lưu `section_path`.
3. Trong mỗi section, gom đoạn văn đến khi đạt khoảng **350–450 token** (đếm bằng tokenizer của embedding model, tối đa 512).
4. **Khối nguyên tử, không bao giờ cắt ngang:**
   - Bảng (dòng bắt đầu bằng `|`, hoặc khối có nhiều dòng chứa tab hay nhiều khoảng trắng thẳng cột).
   - Danh sách liên tục (`-`, `*`, `•`, `1.`, `a)`).
   - Nếu một khối nguyên tử dài hơn giới hạn, để nguyên thành một chunk riêng và đánh dấu `oversized=True`.
5. Overlap khoảng 1 câu (hoặc 50 token) giữa hai chunk liền nhau **trong cùng section**; không overlap qua ranh giới section.
6. Thêm tiêu đề ngữ cảnh vào đầu text khi embed: `"{title} > {' > '.join(section_path)}\n{text}"`.

**Test cần có**

- Bảng liều dùng 3 cột, 10 dòng nằm trọn trong một chunk.
- Văn bản có "Chương I / Điều 1 / Điều 2" cho ra `section_path` đúng.
- Không chunk nào vượt 512 token, trừ chunk `oversized`.

### 5.2 Index

- **Dense:** embed `"passage: " + text_có_tiêu_đề`, chuẩn hóa L2, lưu FAISS `IndexFlatIP`.
- **Sparse:** chữ thường; tiếng Việt tách từ bằng `pyvi`/`underthesea`; tiếng Anh tách theo khoảng trắng + bỏ dấu câu; bỏ stopword cơ bản của cả hai ngôn ngữ. Lưu BM25 bằng pickle.
- `meta.json` lưu: tên model embedding, số chunk, ngày dựng, hash cấu hình chunker. Khi khởi động API, kiểm tra model khớp với index.
- CLI: `python -m medrag.index.build --input data/raw --out data/index`

### 5.3 Hybrid retrieval

```python
def hybrid_search(question: str, k_dense=50, k_bm25=50, k_out=30, rrf_k=60) -> list[ScoredChunk]
```

- Dense: embed `"query: " + question`, lấy top `k_dense`.
- BM25: tách từ câu hỏi cùng cách với index, lấy top `k_bm25`.
- RRF: `score(d) = Σ 1 / (rrf_k + rank_i(d))` trên hai danh sách; lấy top `k_out`.
- Trả về cả rank và điểm từ từng nhánh để tiện debug.

### 5.4 Rerank

```python
def rerank(question: str, chunks: list[ScoredChunk], top_n=5) -> list[ScoredChunk]
```

- Cross-encoder chấm từng cặp (question, chunk text), batch 16–32, chạy GPU nếu có.
- Ngưỡng tùy chọn: bỏ chunk có điểm quá thấp; nếu không còn chunk nào, LLM phải trả lời "không tìm thấy thông tin trong tài liệu".

### 5.5 Backend LLM

```python
class LLMBackend(Protocol):
    name: str
    def generate(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResult: ...
    def stream(self, messages: list[dict], max_tokens: int, temperature: float) -> Iterator[str]: ...

@dataclass
class LLMResult:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_ms: float | None
    total_ms: float
```

- `openai_compat.py` dùng chung cho mọi API tương thích OpenAI; chỉ khác `base_url`, `api_key`, `model`:
  - OpenAI: `https://api.openai.com/v1`
  - vLLM: `http://<host>:8000/v1`
  - Ollama: `http://localhost:11434/v1`
  - Groq: endpoint OpenAI-compatible của Groq
- `gemini.py` bọc SDK Gemini theo cùng interface.
- Đo `ttft_ms` bằng cách luôn gọi ở chế độ streaming nội bộ, kể cả khi API trả kết quả không streaming.

**`.env.example`**

```
LLM_BACKEND=vllm            # openai | vllm | ollama | groq | gemini
LLM_MODEL=<tên model>
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=EMPTY
LLM_MAX_TOKENS=256
LLM_TEMPERATURE=0

EMBED_MODEL=intfloat/multilingual-e5-base
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RETRIEVE_K_DENSE=50
RETRIEVE_K_BM25=50
RETRIEVE_K_OUT=30
RERANK_TOP_N=5
INDEX_DIR=data/index
```

### 5.6 Prompt

- System prompt cố định (đặt **đầu** prompt, giống hệt nhau giữa các request, để tận dụng prefix caching khi benchmark):
  - Chỉ trả lời dựa trên context được cung cấp.
  - Trả lời bằng ngôn ngữ của câu hỏi.
  - Trích dẫn dạng `[1]`, `[2]` ứng với thứ tự context.
  - Không đưa ra chẩn đoán hay liều dùng cho cá nhân; khuyên hỏi ý kiến bác sĩ khi phù hợp.
  - Nếu context không đủ, nói rõ là không tìm thấy.
- Phần user: các context đánh số `[i] {title} > {section_path}\n{text}`, rồi câu hỏi.

### 5.7 Pipeline và đo thời gian

```python
def answer(question: str) -> AnswerResult
# AnswerResult: answer, citations, contexts, timings = {
#   "retrieve_ms", "rerank_ms", "llm_ttft_ms", "llm_total_ms", "total_ms"
# }, backend, model, prompt_tokens, completion_tokens
```

Ghi mỗi request ra `logs/requests.jsonl` (bật/tắt qua `.env`).

---

## 6. API

| Method | Path | Mô tả |
| --- | --- | --- |
| `GET` | `/health` | Trạng thái, backend và model đang dùng, số chunk trong index |
| `POST` | `/ask` | Body `{question, top_n?}` → `{answer, citations[], timings, backend, model}` |
| `POST` | `/ask/stream` | Như trên nhưng trả token qua Server-Sent Events |
| `POST` | `/retrieve` | Chỉ retrieval + rerank, không gọi LLM (để debug và đánh giá) |
| `POST` | `/ingest` | (Tùy chọn) nạp thêm tài liệu và dựng lại index |

---

## 7. Đánh giá

**Bộ dữ liệu**

- Tài liệu: các bài Wikipedia y khoa tiếng Việt và tiếng Anh (bệnh, thuốc, triệu chứng), cộng thêm tài liệu riêng nếu có.
- Bộ QA tổng hợp: `make_synthetic_qa.py` chọn ngẫu nhiên chunk, nhờ LLM sinh 1–2 câu hỏi mà chunk đó trả lời được, lưu `{question, gold_chunk_id, reference_answer, lang}`. Lọc tay khoảng 10% để loại câu hỏi kém. Mục tiêu khoảng **300 câu, cân bằng VI/EN**.

**Chỉ số retrieval** (`eval_retrieval.py`): Recall@5, Recall@10, MRR@10, nDCG@10, báo theo từng ngôn ngữ và cho 4 cấu hình:

1. Chỉ BM25
2. Chỉ dense
3. Hybrid (RRF)
4. Hybrid + rerank

**Chỉ số câu trả lời** (`eval_answers.py`): faithfulness (câu trả lời có bám context không) và answer relevance, chấm bằng LLM-as-judge hoặc `ragas`. Thêm tỷ lệ trả lời "không tìm thấy" đúng lúc, với một nhóm câu hỏi cố ý không có trong tài liệu.

Kết quả lưu ở `eval/results/*.csv` và tóm tắt thành bảng trong README.

---

## 8. Cầu nối sang project benchmark

- `tools/export_prompts.py`: chạy retrieval + rerank cho toàn bộ câu hỏi trong bộ QA, xuất **messages cuối cùng** gửi cho LLM ra `prompts/medical_rag_prompts.jsonl`:
  ```json
  {"id": "q_0001", "lang": "vi", "messages": [...], "approx_input_tokens": 1830}
  ```
  Nhờ vậy mọi backend trong benchmark nhận đầu vào giống hệt nhau, và không phải chạy retrieval lúc đo.
- `timings` trong mỗi response cho thấy LLM chiếm bao nhiêu phần trăm tổng thời gian, một con số rất đáng đưa vào README benchmark.
- Đổi sang vLLM chỉ cần sửa `.env`, không sửa code.

---

## 9. Lộ trình dựng

1. **Khung dự án:** repo, `config.py`, `.env.example`, `pyproject.toml`.
2. **Ingestion + chunker** kèm test (mục 5.1).
3. **Index dense + BM25** và CLI dựng index.
4. **Hybrid search + RRF**, rồi **rerank**; kiểm tra bằng `/retrieve`.
5. **Backend LLM** (bắt đầu với Ollama local), prompt, pipeline có đo thời gian.
6. **FastAPI** với `/ask` và `/ask/stream`.
7. **Bộ QA tổng hợp** và đánh giá retrieval cho 4 cấu hình.
8. **Đánh giá câu trả lời.**
9. **`export_prompts.py`** và thử backend vLLM (khi đã có GPU).
10. **README:** kiến trúc, cách chạy, bảng kết quả.

---

## 10. Tiêu chí hoàn thành

- [ ] `python -m medrag.index.build` dựng được index từ `data/raw` không lỗi.
- [ ] Test chunker pass, kể cả case bảng liều dùng và cấu trúc Chương/Mục/Điều.
- [ ] `/ask` trả lời đúng ngôn ngữ câu hỏi, có trích dẫn `[i]` khớp với `citations`.
- [ ] Hybrid + rerank có Recall@10 cao hơn từng nhánh đơn lẻ trên bộ QA (nếu không, ghi lại và phân tích vì sao).
- [ ] Đổi `LLM_BACKEND` giữa `ollama` và `vllm` chỉ bằng `.env`.
- [ ] `export_prompts.py` sinh file JSONL dùng được cho benchmark.
- [ ] README có bảng kết quả đánh giá và hướng dẫn chạy lại.
