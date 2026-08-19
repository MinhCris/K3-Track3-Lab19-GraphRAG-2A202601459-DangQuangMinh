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

## Ca 1 — Flat RAG thất bại, GraphRAG thành công: **G5000-23** (cross-doc)

- **Câu hỏi:** *"Which Microsoft-related story is a confirmed external customer use of ChatGPT, and which is only a reported potential product for businesses?"*
- **Điểm judge (comprehensiveness):** Flat **1** vs Graph **5** (multi-hop reasoning cũng 1 vs 5).
- **Tại sao Flat RAG thất bại?** Truy vết tầng context: top-6 cosine chỉ gom được bài "Microsoft private ChatGPT" (`art_007711`) — bài trùng từ vựng mạnh với câu hỏi (Microsoft, ChatGPT, businesses) — và **bỏ sót hoàn toàn bài DEWA** (`art_000256`): tiêu đề bài này nói về "DEWA... first utility in the world", gần như không chia sẻ từ vựng với câu hỏi. Flat answer thành thật thú nhận *"confirmed external customer use... is not explicitly mentioned in the provided context"* — tức lỗi nằm ở **retrieval**, không phải generation. Judge rationale xác nhận: *"completely misses the confirmed customer use case (DEWA's February announcement)"*.
- **GraphRAG giải quyết thế nào?** Seed extraction trích Microsoft/ChatGPT → match node trong Neo4j → BFS 2 hop đi theo edge tới node DEWA (quan hệ USES/PARTNERED_WITH quanh ChatGPT/Microsoft), kéo được chunk `art_000256::c0000` vào context. Graph answer trả lời đúng cả 2 vế, dẫn đúng 2 chunk_id (`art_000256` cho vế confirmed, `art_007711` cho vế potential). Đây là cơ chế thắng đặc trưng của GraphRAG: **nối dữ kiện qua thực thể trung gian khi từ vựng không giao nhau**.

## Ca 2 — GraphRAG thất bại: **G5000-30** (multi-hop, cả hai cùng kém — Graph kém hơn)

- **Câu hỏi:** *"Meta appears in two different AI contexts in the selected data. What are they, and what distinct relation should the graph store in each case?"* (Ref: Meta là model-provider của Llama 2/Code Llama trên Google Cloud; và Meta nằm trong danh sách cam kết AI tự nguyện với Nhà Trắng.)
- **Điểm judge (comprehensiveness):** Flat **2** vs Graph **1**.
- **Nguyên nhân (root-cause, dừng ở tầng hỏng đầu tiên — tầng extraction/schema):** Graph answer khẳng định `Google [Company] -PARTNERED_WITH-> Meta [Company]` là một trong hai ngữ cảnh. Edge này có thật trong đồ thị nhưng là **edge quá khái quát**: extraction đã ép ngữ cảnh đồng-xuất-hiện (Llama 2 được đưa lên Google Cloud / hai công ty cùng nằm trong danh sách cam kết) thành PARTNERED_WITH. Sâu hơn: quan hệ đúng mà câu hỏi cần — *model-provider* và *policy-commitment* — **không tồn tại trong 8 loại relation của allowlist**, nên extraction chỉ có thể chọn nhãn gần nhất (sai). Model sinh câu trả lời bám vào edge sai trông-có-vẻ-đúng đó; judge rationale xác nhận: *"states a partnership with Google, which is not explicitly supported"*. Flat RAG cũng fail (2/5) vì bài Google Cloud Next không lọt top-6 — tức câu hỏi này khó ở cả hai kiến trúc, nhưng GraphRAG **tệ hơn vì tự tin sai** dựa trên edge nhiễu — minh họa đúng cảnh báo "false edge nguy hiểm hơn missing edge".
- **Đề xuất khắc phục:** (1) mở rộng allowlist có kiểm soát: thêm `PROVIDES`/`COMMITTED_TO` cho domain tin AI; (2) siết prompt extraction: PARTNERED_WITH chỉ khi văn bản nêu tường minh quan hệ đối tác song phương, không suy từ đồng-xuất-hiện trong danh sách; (3) lọc edge theo `confidence` khi linearize context; (4) prompt trả lời yêu cầu nêu evidence cho từng quan hệ được khẳng định.

---

## Ghi chú thiết kế liên quan (đã cài sẵn để giảm 2 lớp lỗi)

- **Chống thua oan do thiếu dữ liệu:** golden row-id không khớp CSV gốc → pipeline khớp 51 bài evidence theo nội dung `(published_at, title)` và ưu tiên đưa vào corpus + extraction; nếu bỏ qua bước này, phần lớn ca "GraphRAG fail" thực chất là "graph không có dữ kiện" — một lỗi dữ liệu, không phải lỗi kiến trúc.
- **Chống mất tín hiệu cross-doc:** near-dedup có guard `KEEP_EVIDENCE` — hai bản tin gần trùng về cùng sự kiện (vd 2 bài Aeris–Ericsson 12/2022 và 01/2023) chính là tín hiệu cross-doc, không được gộp.
