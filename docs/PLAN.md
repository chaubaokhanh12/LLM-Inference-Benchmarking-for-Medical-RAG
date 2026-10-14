# Kế hoạch thực hiện

Spec gốc: [medical_rag_spec.md](medical_rag_spec.md) (phần 1) và [llm_benchmark_spec.md](llm_benchmark_spec.md) (phần 2).

## Quyết định đã chốt

| Vấn đề | Lựa chọn |
| --- | --- |
| Bố cục | Monorepo 2 phần: `medical-rag/` → `prompts/` (cầu nối) → `inference-bench/` |
| Môi trường | Conda env Python 3.12 tại `D:/envs/medrag` (ổ C gần đầy); torch 2.14 + CUDA 12.6 |
| LLM sinh QA và chấm điểm | OpenAI API (`EVAL_LLM_MODEL`) |
| LLM khi phát triển | Ollama, model nhỏ (dòng Qwen2.5) trên RTX 2050 4 GB |
| vLLM local | Docker Desktop (`vllm/vllm-openai`), model 0.5–1.5B; số liệu benchmark thật chạy trên L4 |
| Bảng quá dài | Giữ nguyên thành 1 chunk `oversized=True` (đúng spec); cờ tách theo dòng để so sánh |
| Dữ liệu | Wikipedia VI/EN (HTML → Markdown để giữ bảng); tài liệu riêng bổ sung sau nếu cần |
| Context tối đa | 4096 token (khớp `max_model_len` của benchmark), prompt được cắt cho vừa |

## Điều chỉnh so với spec (lý do ngắn)

- Heading đánh số `1.` dễ trùng với danh sách đánh số → luật phân xử riêng; `[A-ZÀ-Ỹ]` không kiểm được chữ hoa → `str.isupper()`.
- Số La Mã bắt nhầm "V. cholerae" → yêu cầu chữ hoa phía sau + dòng ngắn.
- Ngân sách 512 token tính trên chuỗi embed thật (`passage: ` + tiêu đề ngữ cảnh + text).
- Cleaner không gộp khoảng trắng trong dòng (giữ cột bảng).
- `stream()` phát sự kiện `Token` / `Done(LLMResult)` để có TTFT và usage khi streaming.
- API chạy cổng 8080 (vLLM dùng 8000).
- Bộ QA lưu thêm `evidence` để ánh xạ lại gold chunk khi đổi chunker.
- Prompt: được nêu liều tham khảo theo tài liệu (có trích dẫn), không cá nhân hóa.

## Lộ trình

- [x] P0 Môi trường, khung dự án
- [x] P1 Thu thập Wikipedia VI/EN (150 EN + 139 VI bài; holdout 42 EN + 19 VI)
- [x] P2 Loader, cleaner, section-aware chunker + test
- [x] P3 Index dense (FAISS) + BM25, CLI build
- [x] P4 Hybrid RRF, rerank, `/health`, `/retrieve`  → **M1**
- [x] P5 Backend LLM, prompt, pipeline đo thời gian (code + test; chạy thật bị giới hạn bởi VRAM/RAM)
- [x] P6 `/ask`, `/ask/stream` (SSE) (code + test)  → **M2**
- [x] P7 Bộ QA tổng hợp: 390 câu (300 có đáp án VI/EN cân bằng, 40 xuyên ngôn ngữ, 50 không có
      đáp án); soát tay mẫu 10% (39/39 giữ)
- [x] P8 Đánh giá retrieval 4 cấu hình (hybrid+rerank R@10 = 1,000, MRR@10 = 0,959) và câu trả lời
      (Ollama qwen2.5:3b); ngưỡng rerank 0,3; thêm câu nhắc trích dẫn cuối prompt  → **M3**
- [x] P9 `export_prompts.py`: 300 prompt, VI p50 1.859 / EN p50 1.666 token (Qwen2.5-7B), `stats.json`, `manifest.json`
- [x] P9b vLLM 0.31.0 Docker local: đổi backend chỉ bằng `.env` (check_backend.py), load test 1/4/8 người dùng (throughput 24 → 188 tok/s); cần driver CUDA 13 (617.42)
- [x] P10 README có bảng kết quả và hướng dẫn chạy lại  → **M4**
- [x] Phần 2 — khung `inference-bench/`: load generator closed-loop, HF baseline server, GPU/vLLM
      metrics, `bench.run`, 14 config E1–E5, aggregate + 6 biểu đồ, quality_eval, test; đã chạy
      thử với Ollama trên laptop
- [x] Phần 2 — notebook Colab `inference-bench/colab/run_benchmark.ipynb` chạy E1–E4 trên L4 (Colab Pro), lưu Drive, tự bỏ qua thí nghiệm đã xong
- [x] Phần 2 — chạy E1–E4 trên Colab L4 (13 cấu hình, 0 lỗi), chấm chất lượng E3, README benchmark trả lời Q1–Q5

## Ràng buộc phần cứng (máy phát triển)

- RTX 2050 4 GB: lúc bị giới hạn công suất (P5, ~16 W) chỉ ~1,3 TFLOPS fp16 và rerank 30 đoạn ≈ 9 s;
  lúc chạy đủ sức (P0, ~29 W) đạt 12,3 TFLOPS, rerank ≈ 1,1 s, dựng toàn bộ index 98 s.
- RAM 7,7 GB: trên Windows (WDDM) bộ nhớ GPU cũng tính vào commit của hệ thống. Khi trình duyệt
  giữ ~7,6 GB commit, nạp embedder + reranker đã đủ làm CUDA báo lỗi cấp phát. Cần đóng bớt ứng
  dụng trước khi chạy. Đánh giá câu trả lời chạy hai pha (`eval_answers.py --two-phase`), Ollama
  dùng biến thể `num_gpu 24`.
- Số đo độ trễ trên máy này chỉ để kiểm tra; số liệu benchmark thật chạy trên GPU L4.
