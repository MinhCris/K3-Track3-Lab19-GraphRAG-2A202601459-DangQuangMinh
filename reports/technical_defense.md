# Thuyết Minh Kỹ Thuật — 10 Câu Hỏi Bảo Vệ Kiến Trúc

**Học viên:** Đặng Quang Minh (2A202601459) · Lab 19 GraphRAG vs Flat RAG · 19/08/2026

> Ô ⬜ điền từ output lần chạy Colab chính thức (nguồn ghi kèm). Chi tiết mở rộng: xem `lab_report.md`.

### 1. Coreference sai ở tình huống nào?
Corpus là `title + description` (~40–50 từ/bài) nên tiền ngữ thường nằm ở title. Tình huống sai điển hình: bản tin nhắc ≥2 công ty ngay trong title ("X and Y Partner to..."), description viết "the company announced..." — tiền ngữ mơ hồ giữa X và Y. Quy tắc conservative yêu cầu giữ nguyên + log `unresolved_mentions`; nếu model vẫn resolve là ca sai tiềm năng, hậu quả là **false edge** gán sự kiện cho nhầm công ty và lan truyền qua BFS. Ví dụ cụ thể từ lần chạy: ⬜ [spot-check ở cell 1.7].

### 2. Entity threshold bao nhiêu, vì sao?
Cosine **0.90** (MiniLM-L6-v2, chuẩn hóa, ANN top-5). Chọn 0.90 vì tên thực thể là chuỗi rất ngắn — ở 0.80–0.90 xuất hiện nhiều cặp "gần nghĩa nhưng khác thực thể" (quan sát được trong audit qua nhãn `REJECT_THRESHOLD`, log từ 0.80 trở lên). Dưới 0.90 rủi ro false merge tăng nhanh hơn lợi ích gộp biến thể; các biến thể phổ biến kiểu hậu tố (Corp/Inc) đã có strip-suffix và manual aliases xử lý trước.

### 3. Candidate nào similarity cao nhưng không nên merge?
⬜ [1 dòng `REJECT_GUARD` similarity cao nhất trong `outputs/entity_resolution_audit.csv`]. Ca điển hình quan sát khi phát triển: `generative AI` vs `Generative AI Solution` (≈0.91) — công nghệ tổng quát vs tên sản phẩm; và bài học ngược: `Amazon Web Services` vs `Amazon Web Services (AWS)` (≈0.93) từng bị guard token-subset chặn **oan**, dẫn tới bổ sung rule initials (token thừa là chữ viết tắt của phần còn lại thì cho merge).

### 4. Top 3 super-node và degree?
⬜ [từ cell 2.4 / `outputs/top_degree_nodes.csv` — 3 hàng đầu: tên, type, degree]. Lưu ý: ở scale lab (400 chunks) không có node degree > 100; policy được kiểm chứng bằng test mô phỏng (`test_supernode_policy_simulated`) với 2 assert: fetch ≤ 50 edge và sort `published_date DESC`.

### 5. Vì sao ưu tiên edge mới nhất có thể đúng/sai?
**Đúng:** tin mới phản ánh trạng thái quan hệ hiện hành (thương vụ hoàn tất thay vì mới đề xuất — đúng tinh thần temporal validity); chặn bùng nổ token tại hub. **Sai:** câu hỏi lịch sử bị cắt mất edge cũ; degree đếm cả 2 chiều nên node "bị nhắc nhiều" chịu cap dù quan hệ chủ động ít; sự kiện lớn nhưng cũ có thể quan trọng hơn tin vụn mới. Khắc phục ở production: cap theo relation type hoặc lọc theo khung thời gian của câu hỏi trước khi áp cap.

### 6. Flat RAG thắng nhóm nào?
⬜ [đọc `outputs/graphrag_vs_flatrag_summary.csv` theo nhóm; kỳ vọng lý thuyết: **factoid** — 1 dữ kiện nằm gọn trong 1 chunk, top-k vector là đủ, rẻ và nhanh hơn]. Nhận xét thực tế: ⬜.

### 7. GraphRAG thắng nhóm nào?
⬜ [tương tự; kỳ vọng lý thuyết: **multi-hop** và **cross-doc** — cần nối dữ kiện qua ≥2 quan hệ hoặc ≥2 bài báo, vector top-k không gom đủ còn BFS theo edge thì có, kèm provenance]. Nhận xét thực tế: ⬜.

### 8. Latency/token trade-off?
Latency đo **end-to-end**: GraphRAG = seed extraction (1 LLM call) + Cypher BFS + generation; Flat = FAISS search + generation. Token: GraphRAG dùng context kép (`=== GRAPH ===` + `=== VECTOR ===`) nên cao hơn rõ rệt. Số cụ thể: ⬜ [hàng Latency/Token trong summary CSV]. Chi phí một lần (indexing) chênh lớn nhất: ~800 LLM calls cho 400 chunks (coref + NER/RE) so với chỉ encode embedding cục bộ của Flat RAG.

### 9. AI Coding Agent đề xuất gì mà bạn **không dùng**, vì sao?
(1) Commit sẵn cache LLM (coref/triples/checkpoint) để lần chấm chạy nhanh — từ chối: bài chấm phải là lần chạy thật; cache chỉ để resume, đã gitignore. (2) Song song hóa vòng eval — từ chối: làm nhiễu số đo latency và phức tạp checkpoint; chỉ song song 2 bước indexing. (3) Near-dedup pairwise O(N²) — từ chối, dùng FAISS ANN O(N·k).

### 10. Scale 350MB: bottleneck đầu tiên là gì?
**LLM extraction**: ~450k bài × 2 lượt gọi (coref + NER/RE) ≈ 900k calls. Giải pháp: worker queue async + batch API; tiền lọc chunk bằng NER thống kê rẻ để giảm số chunk gửi LLM; Entity Resolution chuyển HNSW + blocking; Neo4j giữ `UNWIND` batch + index theo `published_date`; eval bằng sampling + judge phân tầng. Bottleneck kế tiếp mới là Neo4j write throughput và bộ nhớ cho vector index.
