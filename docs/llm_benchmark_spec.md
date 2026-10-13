# LLM Inference Benchmarking cho Medical RAG — Đặc tả

So sánh và tối ưu cách phục vụ (serving) một LLM mã nguồn mở cho workload Medical RAG thật: **HF Transformers** (mốc), **vLLM**, **Ollama**, và tùy chọn **TGI**. Đo TTFT, độ trễ, throughput, bộ nhớ GPU và chất lượng; giải thích chênh lệch bằng số liệu profiling; kết luận nên dùng backend nào cho Medical RAG.

Đầu vào lấy từ project Medical RAG (`tools/export_prompts.py`, xem `medical_rag_spec.md` mục 8).

---

## 1. Câu hỏi cần trả lời

| # | Câu hỏi | Thí nghiệm |
| --- | --- | --- |
| Q1 | Với cùng model và prompt, vLLM nhanh hơn HF Transformers và Ollama bao nhiêu? | E1 |
| Q2 | Khi số người dùng đồng thời tăng từ 1 lên 32, latency và throughput của từng backend thay đổi ra sao? | E2 |
| Q3 | AWQ tiết kiệm bao nhiêu bộ nhớ, đổi lại tốc độ và chất lượng thế nào? | E3 |
| Q4 | Prefix caching giảm TTFT bao nhiêu với system prompt dùng chung của RAG? | E4 |
| Q5 | Vì sao có chênh lệch? (batching, KV cache, mức sử dụng GPU) | Profiling, mục 8 |

---

## 2. Thiết lập cố định

| Thành phần | Giá trị | Lý do |
| --- | --- | --- |
| GPU | 1× NVIDIA L4 24 GB (Google Cloud) | Đủ cho model 7–8B FP16, giá vừa phải |
| Model | Một model instruct 7–8B, hỗ trợ tiếng Việt, có sẵn bản AWQ và GGUF | Cùng một model cho mọi backend; chỉ khác định dạng trọng số khi bắt buộc |
| Prompt | `prompts/medical_rag_prompts.jsonl`, 300 prompt, trộn VI/EN | Workload thật, giống hệt nhau giữa các backend |
| Context tối đa | `max_model_len = 4096` | Đủ cho input 1–3k token + output 256 |
| Output | `max_tokens = 256`, `temperature = 0` | Kết quả ổn định, so sánh được |
| Lặp lại | 3 lần mỗi cấu hình, báo trung vị | Giảm dao động |
| Warm-up | 10 request bỏ qua trước mỗi lần đo | Loại ảnh hưởng của compile, cache lạnh |

Ghi lại vào `results/env.json` ở mỗi lần chạy: tên GPU, driver, CUDA, phiên bản từng thư viện (vllm, transformers, torch, ollama, TGI), tên model và revision. Không có thông tin này thì kết quả không chạy lại được.

---

## 3. Định dạng đầu vào

`prompts/medical_rag_prompts.jsonl`, mỗi dòng:

```json
{"id": "q_0001", "lang": "vi", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "approx_input_tokens": 1830}
```

- System prompt **giống hệt nhau** ở mọi dòng và đặt đầu tiên (cần cho E4).
- Kèm `prompts/stats.json`: phân bố độ dài input (min, p50, p95, max) theo ngôn ngữ, để in trong README.

---

## 4. Các backend

Mọi backend được phục vụ qua HTTP, và (trừ HF) qua API tương thích OpenAI, để load generator chỉ cần một client.

| Backend | Cách chạy | Ghi chú quan trọng |
| --- | --- | --- |
| **HF Transformers** (mốc) | Tự viết `servers/hf_server.py`: FastAPI, endpoint `/v1/chat/completions` tối giản, streaming bằng `TextIteratorStreamer` | Xử lý **từng request một** (dùng lock), không batching động. Đây là "cách làm ngây thơ" để làm mốc |
| **vLLM** | `vllm serve <model> --max-model-len 4096 --gpu-memory-utilization 0.90` | Server tương thích OpenAI ở cổng 8000; có endpoint `/metrics` (Prometheus) dùng cho profiling |
| **vLLM + AWQ** (E3) | Như trên, trỏ tới bản AWQ của model | Kiểm tra cờ quantization theo phiên bản vLLM đang dùng |
| **Ollama** | `ollama serve`, model GGUF tương ứng; gọi qua `/v1` | Đặt `OLLAMA_NUM_PARALLEL` (ví dụ 32) và `OLLAMA_CONTEXT_LENGTH`/`num_ctx` ≥ 4096; mặc định Ollama phục vụ ít request song song và context ngắn, sẽ làm kết quả sai lệch |
| **TGI** (tùy chọn) | Docker image chính thức của TGI | Có API tương thích OpenAI (`/v1/chat/completions`) |

**Lưu ý về so sánh công bằng**

- Ollama dùng GGUF quantized, nên không cùng điều kiện với FP16. Trong README, so Ollama với **vLLM AWQ** (cùng là 4-bit), và ghi rõ mức quantization.
- Prefix caching: phiên bản vLLM mới có thể **bật sẵn** prefix caching. Kiểm tra trong log khởi động, và ở E1–E3 đặt rõ trạng thái bật/tắt để mọi cấu hình giống nhau. Chỉ E4 thay đổi tham số này.
- Cờ dòng lệnh của các thư viện thay đổi theo phiên bản: luôn đối chiếu `--help` của bản đang cài và ghi lệnh thật đã chạy vào `results/<run_id>/command.txt`.

---

## 5. Load generator

`bench/load_gen.py` gửi request đồng thời theo mô hình **closed-loop**: N worker, mỗi worker gửi request kế tiếp ngay khi request trước xong. N chính là "số người dùng đồng thời".

**Hành vi**

1. Đọc prompt từ JSONL, xáo trộn với seed cố định, chia vòng tròn cho N worker.
2. Chạy warm-up 10 request, không ghi.
3. Mỗi request gọi ở chế độ **streaming** (`stream=True`) bằng `openai.AsyncOpenAI` hoặc `httpx`, với `stream_options={"include_usage": true}` khi backend hỗ trợ.
4. Ghi các mốc thời gian: `t_send`, thời điểm nhận chunk có nội dung đầu tiên (`t_first`), thời điểm mọi chunk sau đó, và `t_end`.
5. Lấy số token từ `usage` ở chunk cuối; nếu backend không trả `usage`, đếm lại bằng tokenizer của model.
6. Dừng khi đã chạy đủ số prompt, hoặc hết thời gian tối đa (ví dụ 10 phút).
7. Request lỗi hoặc timeout vẫn được ghi, với `error` khác rỗng.

**Một dòng kết quả** (`results/<run_id>/requests.csv`):

| Cột | Ý nghĩa |
| --- | --- |
| `run_id`, `experiment`, `backend`, `config_name`, `concurrency`, `repeat` | Định danh cấu hình |
| `prompt_id`, `lang`, `prompt_tokens`, `completion_tokens` | Thông tin request |
| `ttft_ms` | `t_first - t_send` |
| `e2e_ms` | `t_end - t_send` |
| `itl_mean_ms`, `itl_p95_ms` | Khoảng cách giữa các chunk liên tiếp |
| `tpot_ms` | `(e2e_ms - ttft_ms) / (completion_tokens - 1)` (time per output token) |
| `error` | Rỗng nếu thành công |

Lưu ý: một chunk streaming không phải lúc nào cũng đúng 1 token, nên dùng `tpot_ms` (tính từ số token thật) làm chỉ số chính, `itl_*` để tham khảo.

---

## 6. Giám sát GPU

`bench/gpu_monitor.py` chạy song song với load generator:

```bash
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw \
           --format=csv,noheader,nounits -lms 1000
```

Lưu `results/<run_id>/gpu.csv`. Với vLLM, đồng thời lấy mẫu `/metrics` mỗi giây và lưu `results/<run_id>/vllm_metrics.csv`; các chỉ số đáng giữ là số request đang chạy, số request đang chờ, và mức sử dụng KV cache. Tên chính xác của metric thay đổi theo phiên bản, nên đọc `/metrics` một lần để lấy đúng tên.

---

## 7. Thí nghiệm

| ID | Mục đích | Backend / cấu hình | Concurrency | Biến thay đổi |
| --- | --- | --- | --- | --- |
| **E1** | So sánh backend ở tải nhẹ | HF, vLLM FP16, vLLM AWQ, Ollama (GGUF) | 1 | Backend |
| **E2** | Khả năng chịu tải | vLLM FP16, vLLM AWQ, Ollama, HF | 1, 4, 8, 16, 32 | Concurrency |
| **E3** | Quantization | vLLM FP16 so với vLLM AWQ | 16 | Định dạng trọng số; đo cả chất lượng |
| **E4** | Prefix caching | vLLM FP16, cache bật / tắt | 16 | Prefix caching |
| E5 (tùy chọn) | So với TGI | TGI FP16 so với vLLM FP16 | 1, 16 | Backend |

Với HF ở E2, concurrency cao sẽ chỉ tạo hàng đợi (vì xử lý tuần tự). Đó chính là kết quả cần cho thấy: TTFT tăng gần tuyến tính theo N, còn throughput không tăng.

**Đánh giá chất lượng (E3 và cho Ollama)**

- Lấy 100 prompt có `reference_answer` từ bộ QA của Medical RAG.
- Chấm bằng LLM-as-judge (faithfulness so với context, độ đúng so với đáp án tham chiếu) với cùng một judge cho mọi cấu hình.
- Báo điểm trung bình và số câu chênh lệch rõ giữa FP16 và AWQ.

**File cấu hình** (`configs/e2_vllm_fp16.yaml`):

```yaml
experiment: E2
backend: vllm
config_name: vllm_fp16
base_url: http://localhost:8000/v1
model: <model id>
server_command: "vllm serve <model> --max-model-len 4096 --gpu-memory-utilization 0.90"
concurrency: [1, 4, 8, 16, 32]
repeats: 3
warmup_requests: 10
max_tokens: 256
temperature: 0
prompts: prompts/medical_rag_prompts.jsonl
timeout_s: 120
```

Một lệnh chạy cả thí nghiệm: `python -m bench.run configs/e2_vllm_fp16.yaml`. Script tự khởi động server, chờ `/health` (hoặc `/v1/models`) sẵn sàng, chạy, rồi tắt server.

---

## 8. Profiling: giải thích vì sao

Báo con số là chưa đủ; README phải giải thích được chênh lệch.

- **Mức sử dụng GPU theo thời gian** (từ `gpu.csv`): HF ở concurrency cao dự kiến có mức sử dụng GPU thấp và bị xếp hàng, còn vLLM giữ GPU bận nhờ continuous batching.
- **Số request chạy song song và mức dùng KV cache** (từ `/metrics` của vLLM): cho thấy lúc nào vLLM chạm giới hạn KV cache và bắt đầu xếp hàng. Đây thường là lý do p95 latency tăng vọt ở concurrency cao.
- **Prefill và decode:** TTFT phản ánh giai đoạn prefill (phụ thuộc độ dài input), `tpot_ms` phản ánh giai đoạn decode. Vẽ TTFT theo `prompt_tokens` để thấy quan hệ này.
- **Ước lượng bộ nhớ KV cache:** tính theo công thức `2 × num_layers × num_kv_heads × head_dim × bytes_per_value × số_token` từ config của model, rồi so với bộ nhớ đo được. Phần này giải thích vì sao AWQ (nhẹ trọng số) cho phép chạy nhiều request đồng thời hơn.
- *(Tùy chọn)* `torch.profiler` trên HF baseline cho vài request, để xem thời gian dồn vào đâu.

---

## 9. Phân tích và biểu đồ

`analysis/aggregate.py` gộp mọi `requests.csv` thành `results/summary.csv`, mỗi dòng là một (backend, config, concurrency):

- p50 / p95 của `ttft_ms`, `tpot_ms`, `e2e_ms`
- Throughput hệ thống = tổng `completion_tokens` / thời gian chạy (không tính warm-up)
- Số request mỗi giây, tỷ lệ lỗi
- Bộ nhớ GPU đỉnh, mức sử dụng GPU trung bình

`analysis/plots.py` vẽ:

1. **Throughput theo concurrency**, một đường mỗi backend (trả lời Q2).
2. **p95 TTFT theo concurrency**, một đường mỗi backend, trục y dạng log (trả lời Q2).
3. **Throughput và p95 latency** trên cùng một hình (đường cong đánh đổi), mỗi điểm là một mức concurrency.
4. **FP16 so với AWQ:** bộ nhớ đỉnh, throughput, điểm chất lượng (trả lời Q3).
5. **TTFT có và không có prefix caching** (trả lời Q4).
6. **TTFT theo số token input** (minh họa prefill, mục 8).

---

## 10. Cấu trúc repo

```
llm-inference-bench/
├── README.md
├── pyproject.toml
├── prompts/
│   ├── medical_rag_prompts.jsonl
│   └── stats.json
├── servers/
│   ├── hf_server.py
│   ├── start_vllm.sh
│   ├── start_ollama.sh
│   └── start_tgi.sh
├── bench/
│   ├── run.py            # đọc config, bật server, chạy, tắt server
│   ├── load_gen.py
│   ├── gpu_monitor.py
│   ├── vllm_metrics.py
│   └── quality_eval.py
├── configs/              # e1_*.yaml ... e4_*.yaml
├── results/              # mỗi run một thư mục: requests.csv, gpu.csv, env.json, command.txt
└── analysis/
    ├── aggregate.py
    └── plots.py
```

---

## 11. Thiết lập Google Cloud

1. **Ngày đầu tiên:** gửi yêu cầu quota GPU L4 cho vùng định dùng; có thể mất 1–2 ngày.
2. Tạo **budget alert** cho dự án.
3. Tạo VM có GPU L4 với image Deep Learning (đã có sẵn driver và CUDA), ổ đĩa ≥ 100 GB để chứa nhiều bản model.
4. Tải trước model vào ổ đĩa (`huggingface-cli download`), tránh tải lại giữa các lần chạy.
5. Chạy thí nghiệm trong `tmux` để không mất tiến trình khi rớt SSH.
6. Sao lưu `results/` về máy hoặc Cloud Storage, rồi **tắt VM** sau mỗi buổi.

Chi phí ước tính: khoảng 10–15 giờ GPU. Tra giá L4 theo vùng trên trang giá của Google Cloud trước khi chạy.

---

## 12. Mẫu README

1. **Tóm tắt kết quả:** 3–4 câu, mỗi câu một con số (ví dụ: "Ở 16 người dùng đồng thời, vLLM đạt throughput gấp X lần HF, p95 TTFT dưới Y ms").
2. **Bối cảnh:** vì sao benchmark cho Medical RAG, mô tả workload và phân bố độ dài prompt.
3. **Thiết lập:** phần cứng, phiên bản thư viện, model, tham số cố định.
4. **Kết quả E1–E4:** mỗi phần một biểu đồ và một bảng ngắn.
5. **Giải thích:** profiling ở mục 8.
6. **Khuyến nghị:** backend và cấu hình nên dùng cho Medical RAG, và khi nào lựa chọn khác hợp lý hơn (ví dụ Ollama cho máy cá nhân không có GPU lớn).
7. **Giới hạn:** một GPU, một model, GGUF và AWQ không hoàn toàn tương đương, v.v.
8. **Cách chạy lại.**

---

## 13. Lộ trình

1. Xuất prompt từ Medical RAG, tạo `stats.json`.
2. Viết `load_gen.py` và `gpu_monitor.py`; thử với Ollama và một model rất nhỏ trên máy cá nhân.
3. Viết `hf_server.py`, `run.py`, các file config.
4. Lên GPU: chạy E1, kiểm tra số liệu hợp lý trước khi chạy tiếp.
5. Chạy E2.
6. Chạy E3 kèm đánh giá chất lượng, rồi E4; lấy số liệu profiling.
7. Tắt GPU. Tổng hợp, vẽ biểu đồ, viết README.
8. Cập nhật Medical RAG dùng vLLM làm backend mặc định nếu kết quả ủng hộ; cập nhật CV với số đo thật.

---

## 14. Tiêu chí hoàn thành

- [ ] Một lệnh chạy lại được bất kỳ thí nghiệm nào từ file config.
- [ ] Mỗi run có `env.json` và `command.txt`.
- [ ] Tỷ lệ lỗi dưới 1% ở mọi cấu hình được báo cáo (nếu cao hơn, ghi rõ và giải thích).
- [ ] Đủ 6 biểu đồ ở mục 9 và bảng `summary.csv`.
- [ ] README trả lời được Q1–Q5, mỗi câu bằng ít nhất một con số.
- [ ] Phần giới hạn và so sánh công bằng được nêu rõ.

---

## 15. Câu hỏi phỏng vấn nên trả lời được

1. PagedAttention và continuous batching giải quyết vấn đề gì của cách serving thông thường?
2. Vì sao tăng concurrency thì throughput tăng nhưng latency từng request cũng tăng? Điểm nào là "đủ tốt"?
3. TTFT phụ thuộc vào prefill thế nào? Vì sao RAG (input dài, output ngắn) nhạy với TTFT?
4. KV cache với context 4.096 token của model bạn chọn chiếm bao nhiêu bộ nhớ, và nó giới hạn số request đồng thời ra sao?
5. AWQ khác GPTQ ở đâu? Khi nào không nên quantize (ví dụ ứng dụng y khoa cần độ chính xác cao)?
6. Prefix caching hoạt động thế nào, và vì sao system prompt phải đặt ở đầu và giữ nguyên?
7. Nếu được thêm 4 GPU, bạn sẽ mở rộng thế nào: tensor parallelism hay nhiều bản sao (replica)? Vì sao?
