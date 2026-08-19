# Reflection Cá Nhân — Đặng Quang Minh (2A202601459)

Lab 19: Production-Grade GraphRAG vs Flat RAG · AICB-K34 Track 3 · 19/08/2026

## 1. Mapping Bài giảng → Code

| Khái niệm bài giảng | Module | Hàm / Khối code | Quan sát thực tế |
|---|---|---|---|
| Conservative Coreference | M1 | `COREF_SYSTEM`, `resolve_coref_batch()`, `run_coref()` | Chunk = title+description rất ngắn nên quy tắc "mơ hồ thì giữ nguyên + log" quan trọng hơn độ phủ; batch lỗi fallback giữ nguyên văn bản |
| Schema & Allowlist Guard | M2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, filter trong `run_extraction()` | Chặn schema drift và chặn chuỗi lạ lọt vào câu Cypher |
| Bulk Cypher Ingestion | M2 | `bulk_insert_nodes()`, `bulk_insert_edges()` (`UNWIND $rows`, batch 1000) | Edge key có `source_chunk_id` nên chạy lại idempotent; `reset_graph()` cho Restart & Run All |
| Entity Resolution + Union-Find | M3 | `build_resolution_map()`, `merge_guard()`, `UF` | Audit 4 nhãn (MERGE_MANUAL/MERGE_VECTOR/REJECT_GUARD/REJECT_THRESHOLD) — nhờ audit mới phát hiện guard chặn oan ca viết tắt (AWS) |
| Super-node Degree Cap | M4 | `retrieve_graph_context()`, `recent_edges()`, `test_supernode_policy_simulated()` | 3 tầng chặn: 50 edge/node · 250 edge tổng · 14000 chars |
| LLM-as-a-Judge | M5 | `judge_answer()`, `run_evaluation()` | Judge khác model generator; có checkpoint/resume |

## 2. Debugging & Bài học

- **Kernel chết không traceback:** notebook chết đúng lúc FAISS chạy op song song đầu tiên sau khi torch encode xong. Truy vết từng lớp: code chạy ngoài notebook thì ổn → nghi môi trường kernel → phát hiện kernelspec `python3` trỏ sai interpreter, và gốc rễ là **2 OpenMP runtime** (libomp của torch + libiomp5 của MKL) cùng nạp. Fix: kernelspec riêng cho venv + `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `faiss.omp_set_num_threads(1)`. **Bài học:** lỗi native không hiện traceback Python — cô lập từng tầng (code → interpreter → runtime) thay vì sửa mò.
- **Khóa nối dữ liệu sai ngầm:** row-id của Golden Dataset đánh trên một frame tiền xử lý khác nên không khớp CSV gốc; nếu tin row-id thì corpus thiếu bài evidence và mọi kết luận benchmark sai từ gốc. Fix: khớp evidence theo nội dung `(published_at, title)`, xác minh 51/51. **Bài học:** kiểm chứng giả định về khóa nối giữa hai nguồn dữ liệu trước khi xây pipeline phía sau.
- **Hạ tầng đổi giữa chừng:** Groq key hết hạn → wrapper `groq_chat` route sang endpoint OpenAI-compatible (OpenRouter) mà không đổi chữ ký hàm; TensorFlow cũ xung đột Keras 3 → `USE_TF=0`. **Bài học:** bọc mọi dependency ngoài (LLM API) sau một interface mỏng để thay provider không phải sửa pipeline.
- **Kiểm soát AI Coding Agent:** các đề xuất bị từ chối và lý do — xem mục 5 của `lab_report.md` (commit cache LLM, song song hóa eval, near-dedup O(N²)). Nguyên tắc rút ra: agent tối ưu cho "chạy được/nhanh", người thiết kế phải giữ ràng buộc "đo được/công bằng/tái lập được".

## 3. Kế hoạch Áp dụng vào Đồ án (Action Plan)

- **Tên đồ án:** ⬜ [cần mô tả đồ án của học viên — sẽ hoàn thiện]
- **Bài toán có cần GraphRAG không?** ⬜ [tiêu chí quyết định rút từ lab: dữ liệu có nhiều quan hệ thực thể chéo tài liệu không; câu hỏi người dùng có dạng multi-hop/cross-doc không; nếu chủ yếu factoid trên tài liệu đơn thì Flat/Hybrid RAG là đủ]
- **Node/Relation dự kiến:** ⬜
- **Chiến lược Entity Resolution & Super-node cho bài toán cụ thể:** ⬜

## 4. Tự đánh giá

| Tiêu chí | Điểm (1–5) | Ghi chú |
|---|---|---|
| Hiểu bài giảng GraphRAG | ⬜ | |
| Kiểm soát AI Coding Agent | ⬜ | |
| Chất lượng knowledge graph | ⬜ | |
| Phân tích & debug hệ thống | ⬜ | |
