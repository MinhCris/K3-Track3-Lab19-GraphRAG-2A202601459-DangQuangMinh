# Thuyết Minh Kỹ Thuật — 10 Câu Hỏi Bảo Vệ Kiến Trúc

**Học viên:** Đặng Quang Minh (2A202601459) · Lab 19 GraphRAG vs Flat RAG · 19/08/2026

> Ô ⬜ điền từ output lần chạy Colab chính thức (nguồn ghi kèm). Chi tiết mở rộng: xem `lab_report.md`.

### 1. Coreference sai ở tình huống nào?
Corpus là `title + description` (~40–50 từ/bài) nên tiền ngữ thường nằm ở title. Tình huống sai điển hình: bản tin nhắc ≥2 công ty ngay trong title ("X and Y Partner to..."), description viết "the company announced..." — tiền ngữ mơ hồ giữa X và Y. Quy tắc conservative yêu cầu giữ nguyên + log `unresolved_mentions`; nếu model vẫn resolve là ca sai tiềm năng, hậu quả là **false edge** gán sự kiện cho nhầm công ty và lan truyền qua BFS. Ví dụ thật từ lần chạy: `art_000399::c0000` — *"Tyber Medical LLC has acquired ADSM-Synchro Medical ... as **the company** expands internationally"* → resolved đúng thành **Tyber Medical** (câu chứa 2 công ty — resolve nhầm sang ADSM-Synchro sẽ sinh false edge). Ca gặp khó: `art_000033::c0000` text bị cắt cụt *"...assets to be"*, model tự hoàn thành *"are to be acquired."* — suy diễn hợp lý từ title nhưng là thêm từ không có trong văn bản. Thống kê: 400 chunks, 17 unresolved mentions/15 chunks; chỉ 3 chunk thực sự resolve generic reference, phần lớn chỉnh sửa còn lại là chuẩn hóa dấu câu.

### 2. Entity threshold bao nhiêu, vì sao?
Cosine **0.90** (MiniLM-L6-v2, chuẩn hóa, ANN top-5). Chọn 0.90 vì tên thực thể là chuỗi rất ngắn — ở 0.80–0.90 xuất hiện nhiều cặp "gần nghĩa nhưng khác thực thể" (quan sát được trong audit qua nhãn `REJECT_THRESHOLD`, log từ 0.80 trở lên). Dưới 0.90 rủi ro false merge tăng nhanh hơn lợi ích gộp biến thể; các biến thể phổ biến kiểu hậu tố (Corp/Inc) đã có strip-suffix và manual aliases xử lý trước.

### 3. Candidate nào similarity cao nhưng không nên merge?
Lần chạy chính thức: audit 14 dòng (11 `REJECT_THRESHOLD` + 3 `MERGE_VECTOR`), guard không phải chặn cặp nào ≥0.85. Ca điển hình quan sát khi phát triển: `generative AI` vs `Generative AI Solution` (≈0.91) — công nghệ tổng quát vs tên sản phẩm; và bài học ngược: `Amazon Web Services` vs `Amazon Web Services (AWS)` (0.9287) từng bị guard token-subset chặn **oan**, dẫn tới bổ sung rule initials — lần chạy chính thức cặp này `MERGE_VECTOR` đúng. Near-miss trung thực trong audit: `L T Technology Services` vs `L&T Technology Services Limited` (0.894) — cùng công ty nhưng dưới ngưỡng 0.90 → false split; cách sửa đúng là manual alias, không phải hạ ngưỡng toàn cục.

### 4. Top 3 super-node và degree?
**Microsoft** (Company, degree 17) · **ServiceNow** (Company, 9) · **OpenAI** (Company, 7) — theo `outputs/top_degree_nodes.csv`. Ở scale lab (400 chunks) không có node degree > 100; policy được kiểm chứng bằng test mô phỏng (`test_supernode_policy_simulated`) với 2 assert: fetch ≤ 50 edge và sort `published_date DESC`. Ở scale 350MB các hub này chắc chắn vượt ngưỡng — phân bố degree dạng power-law đã hiện rõ ngay ở subset nhỏ.

### 5. Vì sao ưu tiên edge mới nhất có thể đúng/sai?
**Đúng:** tin mới phản ánh trạng thái quan hệ hiện hành (thương vụ hoàn tất thay vì mới đề xuất — đúng tinh thần temporal validity); chặn bùng nổ token tại hub. **Sai:** câu hỏi lịch sử bị cắt mất edge cũ; degree đếm cả 2 chiều nên node "bị nhắc nhiều" chịu cap dù quan hệ chủ động ít; sự kiện lớn nhưng cũ có thể quan trọng hơn tin vụn mới. Khắc phục ở production: cap theo relation type hoặc lọc theo khung thời gian của câu hỏi trước khi áp cap.

### 6. Flat RAG thắng nhóm nào?
**Factoid** — đúng kỳ vọng lý thuyết: comprehensiveness hai bên hòa 5.00 nhưng Flat thắng multi-hop reasoning (5.00 vs 4.60), nhanh hơn (1.52s vs 2.54s) và rẻ hơn (628 vs 952 tokens). Với câu 1 dữ kiện nằm gọn trong 1 chunk, context đồ thị là thừa thãi — thậm chí gây nhiễu nhẹ cho judge ở tiêu chí reasoning.

### 7. GraphRAG thắng nhóm nào?
**Cross-doc** rõ rệt nhất: comprehensiveness 3.18 → 4.14 (+0.95, summary tự đánh nhãn "GraphRAG cải thiện rõ"), multi-hop reasoning 3.55 → 4.14. **Multi-hop** cũng thắng: comprehensiveness 3.61 → 4.22 (+0.61). Đúng cơ chế: BFS theo edge gom được dữ kiện nằm rải ở nhiều bài mà top-k cosine bỏ sót (ca G5000-23: Flat bỏ sót bài DEWA, Graph nối Microsoft→ChatGPT→DEWA và ăn 5/5).

### 8. Latency/token trade-off?
Latency đo **end-to-end**: Flat 2.38s vs GraphRAG 4.05s (+70%) — phần chênh chủ yếu là 1 LLM call seed extraction + chuỗi Cypher BFS. Token: 690 vs 1326 (×1.92) do context kép (`=== GRAPH ===` + `=== VECTOR ===`). Đổi lại +0.70 comprehensiveness toàn cục và +0.95 ở cross-doc. Chi phí một lần (indexing) chênh lớn nhất: ~180 LLM calls cho 400 chunks (coref batch 5 + NER/RE batch 4) so với chỉ encode embedding cục bộ của Flat RAG.

### 9. AI Coding Agent đề xuất gì mà bạn **không dùng**, vì sao?
(1) Commit sẵn cache LLM (coref/triples/checkpoint) để lần chấm chạy nhanh — từ chối: bài chấm phải là lần chạy thật; cache chỉ để resume, đã gitignore. (2) Song song hóa vòng eval — từ chối: làm nhiễu số đo latency và phức tạp checkpoint; chỉ song song 2 bước indexing. (3) Near-dedup pairwise O(N²) — từ chối, dùng FAISS ANN O(N·k).

### 10. Scale 350MB: bottleneck đầu tiên là gì?
**LLM extraction**: ~450k bài × 2 lượt gọi (coref + NER/RE) ≈ 900k calls. Giải pháp: worker queue async + batch API; tiền lọc chunk bằng NER thống kê rẻ để giảm số chunk gửi LLM; Entity Resolution chuyển HNSW + blocking; Neo4j giữ `UNWIND` batch + index theo `published_date`; eval bằng sampling + judge phân tầng. Bottleneck kế tiếp mới là Neo4j write throughput và bộ nhớ cho vector index.
