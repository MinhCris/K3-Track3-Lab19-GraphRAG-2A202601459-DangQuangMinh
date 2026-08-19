# Phân Tích Ca Lỗi — Flat RAG vs GraphRAG (Root-Cause Analysis)

**Học viên:** Đặng Quang Minh (2A202601459) · Lab 19 · 19/08/2026

> Ô ⬜ điền từ `outputs/graphrag_eval_results.csv` của lần chạy Colab chính thức.

## Quy trình truy vết nguyên nhân gốc rễ (áp dụng cho từng ca)

Truy ngược theo đúng chiều pipeline, dừng ở tầng đầu tiên bị hỏng:

1. **Judge rationale** (`*_judge_rationale`): judge chê thiếu gì / sai gì?
2. **Answer** (`flat_answer` / `graph_answer`): model trả lời thiếu hay context thiếu?
3. **Context**: với Flat — top-6 chunk có chứa bài evidence không (lỗi *retrieval*)? Với Graph — phần `=== GRAPH ===` có edge liên quan không?
4. **Graph retrieval** (chạy lại `retrieve_graph_context(q, return_debug=True)`): seeds có được trích đúng không → seed có match node trong Neo4j không (exact/fuzzy 0.66) → BFS có thu được edge không → có bị cap cắt không (`supernode_events`, `GLOBAL_EDGE_CAP`)?
5. **Extraction**: dữ kiện có nằm trong 400 chunks được trích xuất không, và NER/RE có bắt được quan hệ đó không (soi `raw_triples_df` theo `source_chunk_id`)?

Snippet chọn ca (chạy trong notebook sau eval):

```python
r = eval_results_df.copy()
r["delta"] = r.graph_comprehensiveness - r.flat_comprehensiveness
display(r.nlargest(3, "delta")[["id","group","delta","question"]])   # ứng viên ca 1
display(r.nsmallest(3, "delta")[["id","group","delta","question"]])  # ứng viên ca 2
```

---

## Ca 1 — Flat RAG thất bại, GraphRAG thành công

- **Question ID & Câu hỏi:** ⬜
- **Điểm judge:** Flat ⬜ vs Graph ⬜ (comprehensiveness) · rationale: ⬜
- **Tại sao Flat RAG thất bại?** ⬜ [điển hình cho nhóm multi-hop/cross-doc: dữ kiện nằm rải ở ≥2 bài; top-6 chunk theo cosine chỉ gom được các bài giống câu hỏi về mặt từ vựng, thiếu mắt xích trung gian]
- **GraphRAG giải quyết thế nào?** ⬜ [liệt kê chuỗi edge thực tế từ phần `=== GRAPH ===`: A -REL→ B -REL→ C kèm date/chunk_id]

## Ca 2 — GraphRAG thất bại hoặc cả hai cùng kém

- **Question ID & Câu hỏi:** ⬜
- **Điểm judge:** Flat ⬜ vs Graph ⬜ · rationale: ⬜
- **Nguyên nhân (theo quy trình trên, dừng ở tầng hỏng đầu tiên):** ⬜ [các nguyên nhân thường gặp đã lường trước trong thiết kế: (a) seed extraction trích cụm quá rộng/hẹp nên không match node; (b) quan hệ cần thiết không thuộc 8 loại allowlist nên extraction bỏ qua; (c) dữ kiện nằm ngoài 400 chunks trích xuất; (d) graph context đúng nhưng judge chấm multi-hop thấp vì câu trả lời không nêu tường minh chuỗi suy luận]
- **Đề xuất khắc phục:** ⬜ [khớp với nguyên nhân: mở rộng aliases/fuzzy threshold; thêm relation type; tăng ngân sách extraction có chọn lọc; prompt trả lời yêu cầu nêu chuỗi suy luận]

---

## Ghi chú thiết kế liên quan (đã cài sẵn để giảm 2 lớp lỗi)

- **Chống thua oan do thiếu dữ liệu:** golden row-id không khớp CSV gốc → pipeline khớp 51 bài evidence theo nội dung `(published_at, title)` và ưu tiên đưa vào corpus + extraction; nếu bỏ qua bước này, phần lớn ca "GraphRAG fail" thực chất là "graph không có dữ kiện" — một lỗi dữ liệu, không phải lỗi kiến trúc.
- **Chống mất tín hiệu cross-doc:** near-dedup có guard `KEEP_EVIDENCE` — hai bản tin gần trùng về cùng sự kiện (vd 2 bài Aeris–Ericsson 12/2022 và 01/2023) chính là tín hiệu cross-doc, không được gộp.
