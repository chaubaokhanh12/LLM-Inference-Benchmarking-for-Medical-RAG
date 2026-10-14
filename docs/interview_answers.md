# Trả lời câu hỏi phỏng vấn (mục 15, llm_benchmark_spec.md)

Mỗi câu trả lời gồm phần khái niệm và số liệu đo được trong project này
(Qwen2.5-7B-Instruct trên 1× NVIDIA L4, 300 prompt Medical RAG; chi tiết:
[inference-bench/README.md](../inference-bench/README.md)).

---

## 1. PagedAttention và continuous batching giải quyết vấn đề gì?

**Vấn đề của cách serving thông thường** (`model.generate()` cho từng request, hoặc static batching):
- Mỗi request được cấp sẵn một vùng KV cache liền mạch cho độ dài tối đa → lãng phí bộ nhớ do đặt
  trước và phân mảnh, nên chứa được ít request đồng thời.
- Batch chỉ được ghép lúc bắt đầu; request mới phải chờ cả batch xong; request ngắn xong sớm vẫn giữ chỗ.
- Bản đơn giản nhất xử lý từng request một: GPU chỉ phục vụ 1 người tại một thời điểm.

**PagedAttention** chia KV cache thành các block cố định (như trang bộ nhớ ảo), cấp phát khi cần và
không cần liền mạch → gần như không lãng phí, và các request có thể dùng chung block (nền tảng của
prefix caching).

**Continuous batching** (lập lịch theo từng bước sinh token): ở mỗi bước, scheduler thêm request mới vào
và bỏ request đã xong ra → GPU luôn có việc, request mới không phải chờ cả batch.

**Số liệu:** HF Transformers (mỗi lần 1 request) giữ nguyên **15 tok/s** từ 1 đến 32 người dùng, p95 TTFT
tăng lên **172 s**; vLLM tăng từ **15,9 lên 114 tok/s (×7,2)** với p95 TTFT **12 s**. Sức chứa KV cache vLLM
báo (86.864 token) đúng bằng bộ nhớ trống chia cho 56 KiB/token, tức không có phần đặt trước cho từng request.

## 2. Vì sao tăng concurrency thì throughput tăng nhưng latency từng request cũng tăng? Điểm nào "đủ tốt"?

- Ở batch 1, sinh mỗi token phải đọc toàn bộ trọng số (~15 GB) → bị giới hạn bởi băng thông bộ nhớ, GPU
  tính toán rất ít. Batch nhiều request dùng chung một lần đọc trọng số → **throughput tăng**.
- Nhưng mỗi bước phải làm nhiều việc hơn (đọc thêm KV cache, attention cho nhiều chuỗi) và còn phải xen
  prefill của request mới → **thời gian mỗi token của từng người tăng**; khi GPU bão hòa, request mới phải
  **xếp hàng chờ prefill** → TTFT tăng nhanh. (Định luật Little: số request đồng thời = throughput × latency.)

**Số liệu (vLLM FP16):** 1 → 32 người dùng: throughput 16 → 114 tok/s, thời gian/token 57 → 240 ms,
p95 TTFT 0,7 → 11,9 s.

**"Đủ tốt"** được quyết định bởi SLO, không phải bởi throughput tối đa. Với mục tiêu p95 TTFT ≤ 2 s, một L4
phục vụ được **khoảng 8 người dùng** (vLLM AWQ: 1,29 request/s). Đó cũng là điểm gãy của đường cong: từ 16
lên 32 người dùng, throughput của AWQ chỉ tăng 3% (114,5 → 118 tok/s) trong khi p95 TTFT tăng gần gấp
3 (5,3 → 14 s).

## 3. TTFT phụ thuộc vào prefill thế nào? Vì sao RAG nhạy với TTFT?

TTFT ≈ thời gian chờ trong hàng đợi + **prefill** (tính attention và MLP cho toàn bộ token của prompt, bị
giới hạn bởi khả năng tính toán) + một bước decode. Ở độ dài 1–3k token, prefill tăng gần như tuyến tính
theo số token đầu vào.

**Số liệu:** TTFT ≈ **0,28 ms × số token đầu vào** trên vLLM FP16 (~3.500 token/s, khoảng 54 TFLOPS trên
121 TFLOPS của L4). Một prompt 1.700 token mất khoảng 0,5 s chỉ cho prefill.

**RAG nhạy với TTFT** vì:
- Prompt dài (context truy xuất), câu trả lời ngắn: **22 token đầu vào cho mỗi token đầu ra** trong
  workload này. Prefill là phần việc chính của GPU.
- Khi có tải, prefill chiếm khoảng **70% thời gian GPU** ở 32 người dùng → request mới phải chờ prefill,
  nên p95 TTFT tăng vọt trong khi KV cache vẫn còn trống (69%).
- Với streaming, TTFT là khoảng người dùng phải nhìn màn hình trống; trong pipeline RAG, retrieval và
  rerank (~1 s) còn cộng thêm vào trước đó.

Đòn bẩy: gửi ít đoạn context hơn hoặc đoạn ngắn hơn (top-3 thay vì top-5), prefix caching cho phần dùng
chung, chunked prefill, và GPU có nhiều sức tính hơn.

## 4. KV cache cho 4.096 token chiếm bao nhiêu, giới hạn số request đồng thời ra sao?

Qwen2.5-7B: 28 layer, 4 KV head (GQA, so với 28 query head), head_dim 128, FP16 (2 byte).

- Mỗi token: 2 (K và V) × 28 × 4 × 128 × 2 byte = **57.344 byte ≈ 56 KiB**.
- Một request 4.096 token: **224 MiB**. (Nếu không có GQA, tức 28 KV head, sẽ gấp 7 lần: khoảng 1,5 GiB.)

Trên L4 với `--gpu-memory-utilization 0.90`:

| | Trọng số | Còn cho KV | Sức chứa | Request 4k đồng thời |
| --- | --- | --- | --- | --- |
| FP16 | 14,3 GiB | 4,64 GiB | 86.864 token | 21 |
| AWQ 4-bit | 5,3 GiB | 12,78 GiB | 239.264 token | 58 |

Số tính tay khớp với log của vLLM (sai lệch dưới 0,1%). Request thật trong workload này dài khoảng 1.800
token → FP16 chứa được khoảng 48 request trước khi hết KV cache; ở 32 người dùng, đo được 69% KV đã dùng.
Khi KV cache đầy, vLLM phải cho request mới chờ hoặc tạm đẩy request đang chạy ra (preemption) → độ trễ
tăng vọt. Đây là lý do AWQ (trọng số nhẹ) phục vụ được nhiều người dùng hơn.

## 5. AWQ khác GPTQ ở đâu? Khi nào không nên quantize?

Cả hai đều là lượng tử hóa sau huấn luyện, chỉ áp dụng cho trọng số (4-bit, activation vẫn FP16), theo nhóm
(group-wise scale), cần một ít dữ liệu hiệu chỉnh.

- **GPTQ:** lượng tử hóa từng layer, từng cột trọng số; dùng thông tin bậc hai (xấp xỉ Hessian từ dữ liệu
  hiệu chỉnh) để điều chỉnh các trọng số còn lại, bù sai số cho cột vừa lượng tử hóa.
- **AWQ (activation-aware):** quan sát rằng khoảng 1% kênh trọng số, ứng với activation lớn, quyết định chất
  lượng; tìm hệ số scale cho từng kênh để bảo vệ các kênh này trước khi lượng tử hóa. Không cần tái tạo
  trọng số, ít phụ thuộc dữ liệu hiệu chỉnh, và có kernel nhanh (Marlin) trong vLLM.

**Số liệu:** AWQ giảm trọng số 63%, sinh token nhanh gấp 3 lần ở 1 người dùng (19 so với 57 ms/token), chất
lượng không đo được khác biệt (correctness 4,67 so với 4,68; khoảng tin cậy 95% chứa 0). Nhưng prefill chậm
hơn 12% (phải giải lượng tử hóa khi GPU đang tính), nên ở 32 người dùng lợi thế chỉ còn 1,03×.

**Khi nào không nên quantize:**
- Khi lỗi nhỏ về số hoặc về sự kiện là nguy hiểm: **liều thuốc**, chỉ số xét nghiệm, tên thuốc hiếm. Bộ QA
  ở đây khá dễ (4,7/5), và vẫn có 4/100 câu chênh từ 2 điểm trở lên → cần kiểm tra riêng trên các câu hỏi
  nặng về con số trước khi dùng cho mục đích y khoa.
- Ngôn ngữ hoặc miền ít xuất hiện trong dữ liệu hiệu chỉnh (ví dụ tiếng Việt chuyên ngành).
- Khi hệ thống không bị giới hạn bởi bộ nhớ: ở tải cao, workload này bị giới hạn bởi tính toán và AWQ gần
  như không giúp gì thêm.
- Khi yêu cầu kiểm định bắt buộc phải dùng đúng model gốc đã được đánh giá.

## 6. Prefix caching hoạt động thế nào? Vì sao system prompt phải ở đầu và giữ nguyên?

vLLM băm (hash) từng block KV theo nội dung token của block đó **cộng với toàn bộ phần trước nó**. Request
mới có phần đầu giống hệt từng token với một request cũ sẽ dùng lại các block đã tính, bỏ qua prefill phần đó.
Block cũ bị loại theo LRU.

Vì hash phụ thuộc vào mọi token đứng trước, **chỉ phần đầu trùng khớp tuyệt đối mới dùng lại được**. Chèn một
mốc thời gian hay ID người dùng vào đầu system prompt sẽ làm mọi block phía sau khác hash, mất hết cache.
Vì vậy system prompt phải đặt **đầu tiên và giống hệt từng byte**; phần thay đổi (context, câu hỏi) đặt sau.
Project này giữ system prompt cố định, đưa chỉ dẫn ngôn ngữ và trích dẫn vào cuối user message, và có test
kiểm tra system prompt giống hệt nhau giữa các request.

**Số liệu (E4, 16 người dùng):** phần dùng chung (system prompt và chat template, ~220/1.700 token) chiếm
13% → TTFT trung vị giảm 11%, p95 giảm 14%, throughput tăng 7%. Thử trên laptop cho thấy một cái bẫy khi
đo: gửi lại đúng các prompt cũ thì 99% token trúng cache và TTFT thấp giả tạo, nên benchmark phải reset
cache trước mỗi lần lặp. Muốn lợi nhiều hơn: chỉ dẫn hoặc ví dụ few-shot dài và dùng chung, hội thoại
nhiều lượt, hoặc sắp xếp context ổn định để các đoạn phổ biến lặp lại ở cùng vị trí.

## 7. Thêm 4 GPU: tensor parallelism hay nhiều replica?

**Chọn replica (data parallel):** 5 instance vLLM độc lập sau một load balancer.
- Model 7B vừa một L4 (AWQ 5,3 GiB, FP16 14,3 GiB), và workload này bị giới hạn bởi **throughput** (nhiều
  người dùng, prefill nặng). Replica tăng năng lực gần tuyến tính: khoảng 5 × 1,3 = **~6,5 request/s** mà
  vẫn giữ p95 TTFT dưới 2 s, không tốn giao tiếp giữa các GPU, và một GPU hỏng không làm sập cả hệ thống.
- L4 nối với nhau qua PCIe (không có NVLink); tensor parallelism phải làm all-reduce ở mỗi layer, nên chi
  phí giao tiếp sẽ ăn mất phần lợi.
- Nên định tuyến **theo prefix** (request có cùng phần đầu đến cùng replica) để giữ tỷ lệ trúng prefix cache.

**Khi nào chọn tensor parallelism:** model không vừa một GPU (ví dụ 32B–70B), hoặc cần giảm độ trễ của
*từng* request (mỗi GPU chỉ đọc 1/N trọng số khi decode, prefill được chia sức tính), ví dụ với prompt rất
dài. Có thể kết hợp: vài replica, mỗi replica dùng TP=2 cho model lớn hơn.
