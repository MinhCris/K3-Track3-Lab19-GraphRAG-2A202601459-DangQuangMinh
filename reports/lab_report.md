# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Đặng Quang Minh (2A202601459)
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026
**Môi trường:** Google Colab + Neo4j AuraDB Free · Generator/Extractor: `openai/gpt-4o-mini` (OpenRouter, fallback do Groq key hết hạn) · Judge: `google/gemini-2.5-flash` (cố ý khác model sinh để giảm self-preference bias)

> **Quy ước:** các ô đánh dấu ⬜ được điền từ output của **lần chạy Colab chính thức** (nguồn ghi kèm từng ô). Mọi nội dung còn lại là quyết định thiết kế và sự kiện có thật trong quá trình làm bài.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

*Trả lời:*
- **Đặc thù dữ liệu:** mỗi bản ghi của corpus (mirror `MongoDB/tech-news-embeddings`) chỉ gồm `title + description` (~40–50 từ), nên mỗi bài = 1 chunk. Đại từ/cụm generic (`the company`, `it`, `they`) xuất hiện chủ yếu trong `description` và tiền ngữ thường nằm ở `title`.
- **Ví dụ từ dữ liệu (chạy thật):** chunk `art_000399::c0000` — *"Tyber Medical LLC has acquired ADSM-Synchro Medical a French company for an undisclosed price as **the company** expands internationally"* → resolved thành *"...as **Tyber Medical** expands internationally"*. Đây đúng kiểu ca mơ hồ nguy hiểm: câu chứa 2 công ty (Tyber Medical và ADSM-Synchro), "the company" resolve sai sang ADSM-Synchro sẽ tạo false edge về việc công ty Pháp này "mở rộng quốc tế".
- **Ca gặp khó khăn thực tế:** chunk `art_000033::c0000` (bài Aeris–Ericsson) — text gốc bị cắt cụt *"...and related assets **to be**"*, model đã tự hoàn thành thành *"...are to be **acquired**."*. Suy diễn này hợp lý (title là "Aeris to Acquire IoT Business from Ericsson") nhưng về nguyên tắc là thêm từ không có trong văn bản — đúng ranh giới mà quy tắc conservative phải canh chừng.
- **Thống kê lần chạy:** 400 chunks qua coref; 359 chunk có chỉnh sửa nhưng soi diff thì phần lớn chỉ là chuẩn hóa dấu câu/lỗi escape (`''`→`'`) — chỉ 3 chunk thực sự phân giải generic reference ("the company"/"the firm"); 17 mentions trên 15 chunks được log vào `unresolved_mentions`. Quan sát đáng giá: LLM có xu hướng "tiện tay" copy-edit vượt mandate dù prompt chỉ yêu cầu resolve — một lý do nữa để giữ quy tắc conservative và audit diff.
- **Hậu quả đối với Graph:** phân giải sai chủ ngữ ⇒ NER+RE trích ra **false edge** gán sự kiện (ACQUIRED / INVESTED_IN...) cho nhầm công ty. False edge nguy hiểm hơn missing edge vì BFS traversal sẽ lan truyền dữ kiện sai vào context của mọi câu hỏi đi qua node đó, kèm cả "evidence" trông có vẻ hợp lệ.
- **Biện pháp đã cài đặt:** prompt conservative (chỉ resolve khi tiền ngữ rõ trong cùng chunk, cấm bịa, giữ nguyên số/ngày/ticker), log `unresolved_mentions`, batch nào lỗi thì fallback giữ nguyên văn bản gốc và đánh dấu `COREF_BATCH_FAILED` thay vì làm hỏng dữ liệu.

---

### 2. Entity Resolution Threshold & Lexical Guard

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.90` cho quyết định merge (ANN top-5, embedding MiniLM-L6-v2 chuẩn hóa). Ngoài ra mọi cặp candidate có similarity ≥ `0.80` đều được ghi vào audit với nhãn `REJECT_THRESHOLD` để hậu kiểm near-miss — audit vì thế luôn minh bạch và đủ dày.
- **Cặp bị Guard chặn (similarity > 0.85):** lần chạy chính thức audit có 14 dòng (11 `REJECT_THRESHOLD`, 3 `MERGE_VECTOR`) và guard **không phải chặn cặp nào ≥ 0.85** — nhưng chính điều đó là kết quả của một vòng lặp thiết kế có bằng chứng: ở lần chạy thử khi phát triển, guard token-subset đã chặn `Amazon Web Services` vs `Amazon Web Services (AWS)` (cosine 0.9287) — một **false reject** vì "AWS" chỉ là viết tắt; sau khi bổ sung rule initials, lần chạy chính thức cặp này được `MERGE_VECTOR` đúng (dòng hiện hữu trong audit). Lần chạy thử cũng ghi nhận guard chặn đúng `generative AI` vs `Generative AI Solution` (≈ 0.91) — công nghệ tổng quát vs tên sản phẩm.
- **Near-miss đáng chú ý (từ nhãn `REJECT_THRESHOLD`):** `L T Technology Services` vs `L&T Technology Services Limited` (0.894) — cùng một công ty nhưng nằm **dưới** ngưỡng 0.90 nên không merge, dẫn đến false split (cả hai xuất hiện tách node trong đồ thị). Đây là trade-off trung thực của ngưỡng 0.90: giảm false merge nhưng chịu vài false split; cách xử lý đúng là thêm manual alias cho các biến thể `&`/`Ltd.` thay vì hạ ngưỡng toàn cục.
- **Cơ chế guard (thiết kế theo loại thực thể):**
  - `Person`: họ (token cuối) phải trùng tuyệt đối, tên phải là prefix/viết tắt của nhau ⇒ chặn *Sam Altman* vs *Steve Altman* dù vector similarity rất cao.
  - `Company/Technology`: nếu tên A là token-subset thực sự của tên B và phần thừa không phải hậu tố doanh nghiệp (Inc/Corp/Platforms...) ⇒ nghi là product/sub-brand, từ chối merge (*Apple* vs *Apple Watch*, *Microsoft* vs *Microsoft Teams*), nhưng *Meta* vs *Meta Platforms* vẫn merge nhờ strip suffix.
  - **Ngoại lệ học từ dữ liệu thật:** guard token-subset ban đầu chặn oan `Amazon Web Services` vs `Amazon Web Services (AWS)` (cosine ≈ 0.93) — token thừa "AWS" chính là chữ viết tắt. Đã bổ sung rule: token thừa duy nhất == initials của phần còn lại ⇒ cho phép merge. Đây là ví dụ điển hình vì sao audit table quan trọng: không có audit sẽ không phát hiện false-reject này.

---

### 3. Đồ thị & Super-node Mitigation

*Trả lời:*
- **Top 3 Super-nodes** (nguồn: cell 2.4 / `outputs/top_degree_nodes.csv`):

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | Microsoft | Company | 17 |
| 2 | ServiceNow | Company | 9 |
| 3 | OpenAI | Company | 7 |

- **Lưu ý trung thực về quy mô:** với scale guard 400 chunks trích xuất, đồ thị lab không tạo ra node degree > 100, nên policy được kiểm chứng bằng 2 cách: (1) `test_supernode_policy()` trên node bậc cao nhất, (2) `test_supernode_policy_simulated()` ép node bậc cao nhất qua cap và assert 2 bất biến: số edge trả về ≤ 50 và sort đúng `published_date DESC`. Ở scale thật (350MB) các hub như Microsoft/Google chắc chắn vượt 100.
- **Ưu điểm của temporal cap (50 edge mới nhất):** chặn bùng nổ context/token tại hub; tin mới thường phản ánh trạng thái quan hệ hiện hành (deal đã đóng thay vì mới công bố); kết hợp `GLOBAL_EDGE_CAP = 250` và `MAX_GRAPH_CONTEXT_CHARS = 14000` tạo 3 tầng chặn.
- **Rủi ro:** câu hỏi lịch sử ("thương vụ năm 2021...") có thể bị cắt mất edge cũ tại hub; degree tính cả 2 chiều nên node bị nhiều nguồn nhắc tới (được trỏ đến nhiều) chịu cap dù edge chủ động ít. Hướng khắc phục ở production: cap theo *relation type* (giữ tối thiểu k edge mỗi loại quan hệ) hoặc lọc theo khung thời gian suy ra từ câu hỏi trước khi áp cap.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

Golden Dataset: **50 câu** (5 factoid · 23 multi-hop · 22 cross-doc) từ `data/graphrag_golden_50_first5000.csv`, đầy đủ `reference_answer`. Lưu ý kỹ thuật quan trọng: row-id trong file golden được đánh trên một bản tiền xử lý khác của dataset, nên pipeline khớp 51 bài evidence **theo nội dung (published_at + title)** — đã xác minh đủ 51/51 — và ưu tiên đưa các bài này vào corpus 1500 bài + ngân sách 400 chunks trích xuất.

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge) — nguồn: `outputs/graphrag_vs_flatrag_summary.csv`, hàng "TẤT CẢ":

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Độ chênh lệch (Δ) | Nhận xét phân tích |
|-------------------|----------|----------|--------------------------|-------------------|
| **Comprehensiveness (1–5)** | 3.56 | 4.26 | **+0.70** | Chênh lớn nhất ở cross-doc: 3.18 → 4.14 (+0.95, "GraphRAG cải thiện rõ") và multi-hop: 3.61 → 4.22 (+0.61); factoid hòa 5.00 |
| **Faithfulness (1–5)** | 4.08 | 4.42 | +0.34 | Context đồ thị kèm provenance giúp câu trả lời bám dẫn chứng hơn |
| **Multi-hop Reasoning (1–5)** | 3.92 | 4.22 | +0.30 | Riêng factoid GraphRAG thấp hơn (4.60 vs 5.00) — context đồ thị thừa thãi cho câu 1 dữ kiện |
| **Latency trung bình (s)** | 2.38 | 4.05 | +1.67s | Đo end-to-end: GraphRAG gồm seed extraction (1 LLM call) + BFS Neo4j + generation |
| **Token usage trung bình** | 690 | 1326 | ×1.92 | GraphRAG dùng context kép (`=== GRAPH ===` + `=== VECTOR ===`) |

#### Phân tích 2 Ca lỗi Điển hình (root-cause đầy đủ trong `reports/failure_analysis.md`):
1. **Ca Flat RAG thất bại (GraphRAG thành công): G5000-23** (cross-doc) — comprehensiveness Flat **1** vs Graph **5**. Câu hỏi phân biệt 2 bản tin Microsoft/ChatGPT; Flat RAG chỉ retrieve được bài "private ChatGPT" (gần câu hỏi về từ vựng) và tuyên bố "không thấy confirmed use case", trong khi bài DEWA dùng ChatGPT bị top-6 cosine bỏ sót. GraphRAG đi từ seed Microsoft/ChatGPT theo edge tới node DEWA và trả lời đúng cả 2 vế kèm chunk_id.
2. **Ca GraphRAG thất bại: G5000-30** (multi-hop) — comprehensiveness Flat 2 vs Graph **1**. Đồ thị chứa edge `Google -PARTNERED_WITH-> Meta` được trích từ ngữ cảnh đồng-xuất-hiện (Llama 2 lên Google Cloud / danh sách cam kết AI), model bám vào edge sai này thay vì quan hệ "model-provider" — vốn **không tồn tại trong 8 loại allowlist**. Lỗi kép: extraction ép co-mention thành PARTNERED_WITH + schema thiếu loại quan hệ phù hợp.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:** GraphRAG trả giá bằng (1) chi phí indexing một lần rất lớn — mỗi chunk qua 2 lượt LLM (coref + NER/RE), tức ~800 LLM calls cho 400 chunks, trong khi Flat RAG chỉ cần encode embedding cục bộ; (2) mỗi query thêm 1 LLM call (seed extraction) + các truy vấn Cypher; (3) token/query cao hơn do context kép. Đổi lại, GraphRAG nối được dữ kiện nằm rải rác ở nhiều bài (multi-hop/cross-doc) mà top-k vector không gom đủ, và mỗi dữ kiện đều mang provenance (`chunk_id`, `published_date`, `evidence`) kiểm chứng được. Với câu factoid đơn giản, Flat RAG là đủ và rẻ hơn — đúng như kỳ vọng của bài giảng.
- **Đề xuất của AI Coding Agent mà tôi từ chối:**
  1. *Commit sẵn cache LLM (coref/triples/eval-checkpoint) vào repo để lần chạy chấm bài "ăn" cache cho nhanh, rẻ* — từ chối: bài chấm phải là một lần chạy thật end-to-end; cache chỉ dùng làm cơ chế resume nội bộ khi bị ngắt giữa chừng, và đã đưa vào `.gitignore`.
  2. *Chạy song song vòng eval 50 câu (ThreadPool) để tiết kiệm thời gian* — từ chối: số đo latency per-query sẽ bị nhiễu do tranh chấp tài nguyên/rate-limit, và checkpoint theo thứ tự sẽ phức tạp; chỉ song song hóa 2 bước indexing (coref, extraction) là nơi latency không phải số liệu báo cáo.
  3. *Near-dedup bằng pairwise cosine toàn dataset* — từ chối vì O(N²); thay bằng FAISS ANN top-k, O(N·k) (đúng ràng buộc Challenge A).
- **Giải pháp khi scale 350MB (~450k bài):** bottleneck đầu tiên là **LLM extraction** (450k chunks × 2 calls ≈ 900k calls — vài nghìn USD và nhiều ngày nếu chạy tuần tự). Kiến trúc đề xuất: (1) hàng đợi worker async (queue + retry + dead-letter) cho coref/NER-RE, chạy batch qua API batch giá rẻ; (2) lọc trước bằng heuristic rẻ (NER thống kê/spaCy) để chỉ gửi LLM những chunk có ≥2 thực thể tiềm năng; (3) Entity Resolution chuyển sang HNSW + blocking theo ký tự đầu/loại thực thể thay vì IndexFlat; (4) Neo4j: giữ `UNWIND` batch nhưng tách transaction theo relation type, thêm index phụ theo `published_date` cho truy vấn temporal cap; (5) chấm điểm bằng sampling + judge theo tầng (model rẻ lọc, model đắt phân xử ca sát nút).

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `COREF_SYSTEM`, `resolve_coref_batch()`, `run_coref()` | Chunk ngắn (title+description) nên tiền ngữ thường nằm trong title; quy tắc "ambiguous thì giữ nguyên + log" quan trọng hơn độ phủ |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, filter trong `run_extraction()` | Relation ngoài allowlist bị loại trước khi chạm Cypher — vừa chống schema drift vừa chống injection vào câu lệnh |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` với `UNWIND $rows`, batch 1000 | Edge key gồm `source_chunk_id` nên re-run idempotent; `reset_graph()` đảm bảo Restart & Run All sạch |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `merge_guard()`, class `UF` | Guard theo loại thực thể + audit 3 nhãn (`MERGE_*`/`REJECT_*`) giúp phát hiện cả false-merge lẫn false-reject (ca AWS) |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `recent_edges()`, `test_supernode_policy_simulated()` | 3 tầng chặn: cap 50/node · `GLOBAL_EDGE_CAP=250` · `MAX_GRAPH_CONTEXT_CHARS=14000` |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `run_evaluation()` + checkpoint/resume | Judge dùng model khác generator; chấm 3 tiêu chí kèm rationale; checkpoint cho phép resume khi đứt giữa chừng |

### 2. Quá trình Debugging & Bài học
- **Lỗi khó nhất:** Jupyter kernel **chết không thông báo lỗi** (DeadKernelError) đúng tại thời điểm FAISS chạy op song song đầu tiên ngay sau khi torch encode xong. Truy vết: chạy lại từng đoạn ngoài notebook thì bình thường ⇒ nghi khác biệt môi trường kernel; phát hiện kernelspec `python3` trỏ sang một Python khác (anaconda) chứ không phải venv của dự án, và bản chất lỗi là **hai OpenMP runtime** (libomp của torch + libiomp5 của MKL) cùng nạp vào một tiến trình. Xử lý: đăng ký kernelspec riêng trỏ đúng venv + set `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `faiss.omp_set_num_threads(1)` ngay đầu notebook. Bài học: segfault ở thư viện native không hiện traceback Python — phải nghi ngờ tầng môi trường trước khi nghi code.
- **Lỗi dữ liệu "ngầm" quan trọng nhất:** row-id trong Golden Dataset không khớp CSV gốc (golden được sinh trên một frame đã tiền xử lý khác). Nếu cứ tin row-id, corpus sẽ **thiếu bài evidence** và GraphRAG thua oan. Xử lý: parse `(published_at, title)` từ `reference_evidence` và khớp theo nội dung — xác minh đủ 51/51 bài trong ~8.5k dòng đầu. Bài học: luôn audit giả định về khóa nối giữa 2 nguồn dữ liệu trước khi build pipeline phía sau.
- **Sự cố vận hành:** Groq API key hết hạn giữa chừng ⇒ viết wrapper route sang endpoint OpenAI-compatible (OpenRouter) mà không đổi interface các hàm phía sau (`groq_chat` giữ nguyên chữ ký); TensorFlow cũ trong môi trường xung đột Keras 3 ⇒ `USE_TF=0` ép transformers chỉ dùng PyTorch.

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Hệ thống hỏi–đáp tin tức & hồ sơ doanh nghiệp công nghệ — phát triển tiếp pipeline Production RAG đã xây ở Day 18 (Qdrant + reranking) thành **Hybrid RAG có tầng đồ thị**.
- **Đặc thù bài toán & Lý do chọn giải pháp:** áp tiêu chí rút từ lab: phần lớn truy vấn người dùng là factoid trên 1 tài liệu → Flat RAG (Qdrant) vẫn là nền, vì lab cho thấy factoid hai bên hòa 5.00 nhưng Flat rẻ hơn ~2× token và nhanh hơn ~1.7×. Tầng GraphRAG chỉ bật cho câu hỏi quan hệ (thâu tóm, đầu tư, nhân sự giữa các công ty) — nơi lab đo được cải thiện +0.95 comprehensiveness ở cross-doc. Router phân loại câu hỏi (factoid vs relational) quyết định tầng retrieval, tránh trả phí đồ thị cho mọi truy vấn.
- **Cấu trúc Node & Relation dự kiến:** Nodes `Company`, `Person`, `Product/Technology` (base `Entity`); Relations kế thừa 8 loại của lab **bổ sung `PROVIDES` và `COMMITTED_TO`** — đúng bài học từ ca lỗi G5000-30, nơi allowlist thiếu loại quan hệ "model-provider" khiến extraction ép co-mention thành PARTNERED_WITH sai.
- **Chiến lược xử lý Super-node & Entity Resolution:** ER giữ pipeline 4 tầng của lab (manual aliases cho ticker/biến thể `&`–`Ltd.` học từ ca false-split L&T → ANN 0.90 → guard theo loại thực thể có rule initials → Union-Find) kèm audit table bắt buộc trong CI dữ liệu; Super-node dùng temporal cap 50 nhưng nâng cấp thành **cap theo từng relation type** (giữ tối thiểu k edge mỗi loại) để câu hỏi lịch sử không bị tin mới đè mất.

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Nắm cơ chế retrieval và trade-off của từng nhóm câu hỏi; chứng minh qua phân tích root-cause 2 ca lỗi thay vì chỉ đọc số |
| Khả năng kiểm soát AI Coding Agent | 5 | Từ chối 3 đề xuất của agent có lý do kỹ thuật; phát hiện guard chặn oan (ca AWS) nhờ yêu cầu audit table thay vì tin kết quả |
| Chất lượng đồ thị tri thức xây dựng | 4 | 100% edge có provenance, audit minh bạch; tự trừ 1 vì còn false split (L&T) và edge PARTNERED_WITH quá khái quát ở ca G5000-30 |
| Khả năng phân tích và debug hệ thống | 5 | Truy vết được kernel chết không traceback (2 OpenMP runtime), phát hiện golden row-id lệch và giải bằng khớp nội dung |
