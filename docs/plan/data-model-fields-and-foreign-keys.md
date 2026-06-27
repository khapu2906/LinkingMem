# Data Model — Fields & Foreign Key Analysis

Phân tích cấu trúc dữ liệu hiện tại: schema các trường, khoá ngoại, index, và các gaps cần giải quyết.

---

## 1. Schema hiện tại

### Node (`NodeInfo`)

| Trường | Kiểu | Vai trò |
|--------|------|---------|
| `id` | `u32` | Khoá chính (8-bit instance_id + 24-bit counter) |
| `name` | `String` | Tên hiển thị / entity label |
| `node_type` | `String` | Loại entity — do người dùng định nghĩa, không có enum cố định |
| `weight` | `f32` | Điểm quan trọng tổng hợp `(in_degree + out_degree) / max_combined`, 0.0–1.0 |
| `props` | `serde_json::Value` | Metadata tuỳ ý (JSON), không giới hạn trường |
| `full_context` | `String` | Mô tả đầy đủ — truyền vào LLM khi sinh câu trả lời |
| `embed_context` | `Option<String>` | Text rút gọn cho HNSW embedding; fallback về `full_context` → `name` nếu thiếu |
| `external_id` | `String` | ID ổn định công khai — user-provided hoặc tự sinh `"{node_type}:{name}"`. Không bao giờ thay đổi qua các lần merge. Rỗng với nodes từ snapshot cũ (pre-v0.2.0). |

### Edge (`EdgeInfo`)

| Trường | Kiểu | Vai trò |
|--------|------|---------|
| `from` | `u32` | **Khoá ngoại** tham chiếu `NodeInfo.id` (node nguồn) |
| `to` | `u32` | **Khoá ngoại** tham chiếu `NodeInfo.id` (node đích) |
| `edge_type` | `String` | Nhãn quan hệ — do người dùng định nghĩa (vd. `"works_at"`, `"manages"`) |
| `weight` | `f32` | Confidence / độ mạnh quan hệ (0.0–1.0), dùng trong BFS scoring |
| `full_context` | `String` | Mô tả đầy đủ quan hệ — truyền vào LLM |
| `embed_context` | `Option<String>` | Text rút gọn cho edge HNSW embedding; fallback về `full_context` → `edge_type` |
| `edge_id` | `u64` | Monotonic edge ID được cấp phát tại ingest. Unique trong một process lifetime, không stable qua merge (edge_ids.json chứa giá trị sau merge, tạm thời là 0). 0 với edges từ snapshot cũ. |

---

## 2. Khoá ngoại & Tham chiếu chéo

### ID Allocation (phân tán)

```
node_id (u32) = [instance_id: 8 bit] | [local_counter: 24 bit]
```

- `instance_id`: set qua env var `INSTANCE_ID` (0–255)
- `local_counter`: tăng dần per-instance, tối đa 16M nodes/instance
- 256 instances × 16M nodes = ~4 tỷ nodes tổng

### external_id (stable)

`external_id` được set tại ingest và không bao giờ thay đổi:
- `/ingest/text`: lấy từ `entity.id` trong LLM extraction payload; fallback về `"{node_type}:{name}"` nếu rỗng.
- `/ingest/json`: lấy từ `entity.id`; fallback tương tự.
- Nodes từ snapshot cũ có `external_id = ""` và không được index.

`CsrGraph` build in-memory `external_id_index: HashMap<String, u32>` trong `build()` và expose qua `get_by_external_id()`.

### Khoá ngoại trong Edge

- `edge.from` và `edge.to` là `u32` trỏ trực tiếp vào `NodeInfo.id` (CSR index)
- Edge HNSW lưu `edge_endpoints: Vec<(u32, u32)>` để map edge index → `(from_id, to_id)` khi BFS seed
- Ingest: string ID trong payload được map sang `u32` qua `HashMap<String, u32>` trong `graph/builder.rs`

### Parallel Array Design (storage)

Tất cả metadata edge được lưu dưới dạng parallel arrays, indexed theo vị trí edge trong CSR:

| File | Format | Index |
|------|--------|-------|
| `nodes.json` | JSON array (chứa `external_id`) | `node_id` |
| `edges.bin` | Binary CSR (`u32`) | vị trí edge trong CSR |
| `edge_types.json` | JSON array | vị trí edge trong CSR |
| `edge_contexts.json` | JSON array | vị trí edge trong CSR |
| `edge_embed_contexts.json` | JSON array | vị trí edge trong CSR |
| `edge_ids.json` | JSON array `[u64]` | vị trí edge trong CSR |
| `vectors.bin` | Binary mmap | `node_id` (zero-copy) |
| `edge_vectors.bin` | Binary mmap | vị trí edge trong CSR |
| `edge_endpoints.json` | JSON array `[(from, to)]` | vị trí edge trong CSR |

---

## 3. Index & Lookup

| Index | Vị trí | Phức tạp | Mục đích |
|-------|--------|----------|---------|
| CSR forward | `graph/csr.rs` | O(1) | Lấy neighbours của node |
| CSR reverse | `graph/csr.rs` | O(1) | BFS ngược chiều |
| Node HNSW | `vector/hnsw.rs` | O(log n) | ANN search theo embedding |
| Edge HNSW | `vector/hnsw.rs` | O(log n) | ANN search theo embedding quan hệ |
| VectorStore mmap | `vector/store.rs` | O(1) | Zero-copy truy cập vector theo `node_id` |
| DeltaGraph adj | `delta.rs` | O(k) | Write buffer, linear scan |
| Query cache | `query.rs` (moka LRU) | O(1) | Memoize kết quả query (TTL 300s) |

---

## 4. Gaps & Hạn chế

### 4.1 Không có FK constraint

- Edge có thể tham chiếu `node_id` không tồn tại (orphaned edge)
- Không có validation referential integrity khi ingest
- Xoá node không tự xoá edge liên quan

**Hướng xử lý**: Thêm bước validate `from`/`to` tồn tại trong `graph/builder.rs` trước khi ghi vào DeltaGraph.

### 4.2 `name` không unique

- Trùng tên entity tạo node khác nhau nếu vượt ngưỡng cosine similarity (mặc định 0.92)
- Entity resolution chỉ dựa trên vector similarity, không có LLM-based disambiguation (Phase 10+)

### 4.3 Không có secondary index trên `props`

- `props` là JSON tuỳ ý — filter theo trường trong `props` (vd. `props.company_id`) cần full scan
- Endpoint `/nodes?type=Person` hiện cũng cần full scan

**Hướng xử lý**: Thêm in-memory `HashMap<String, Vec<u32>>` index theo `node_type` và các key hay dùng trong `props`.

### 4.4 Không có multi-tenancy / document isolation

- Toàn bộ graph là global — mọi query tìm trên toàn bộ nodes/edges
- Không phân vùng theo document, collection, hay tenant

**Hướng xử lý**: Thêm trường `collection_id: Option<String>` vào `NodeInfo`/`EdgeInfo` + filter trong BFS/HNSW.

### 4.5 ID không thể tái sử dụng

- Sau 16M nodes/instance, cấp phát thất bại — không có compaction/tombstoning
- Không có versioning cho embedding (đổi model → rebuild toàn bộ)

### 4.6 Merge ID reassignment (fixed in v0.2.0)

**Vấn đề**: `alloc_node_id()` trả về `(instance_id << 24) | local_counter`. Với `INSTANCE_ID=1`, ID đầu tiên là `16_777_216`. Trong `merge_into()`, ID này bị ghi đè bằng `base_count + i` (ví dụ `104`). Tuy nhiên `delta.new_edges` vẫn tham chiếu ID cũ `16_777_216`, khiến `CsrGraph::build()` index out-of-bounds khi build CSR offset arrays — **silent data corruption trong distributed mode**.

**Fix**: `merge_into()` nay build `HashMap<u32, u32>` mapping `old_delta_id → new_csr_index` trước khi xử lý edges, remap tất cả `from`/`to` của delta edges qua map này. Với single-instance (INSTANCE_ID=0), alloc'd IDs đã khớp với sequential reassignment nên remap là no-op.

---

## 5. Tóm tắt

| Khía cạnh | Trạng thái hiện tại |
|-----------|---------------------|
| Schema node/edge | Linh hoạt — `node_type` tuỳ ý, `props` JSON không giới hạn |
| Khoá ngoại | Có (numeric `u32`) — nhưng không có constraint enforcement |
| Primary index | CSR bidirectional — O(1) neighbour lookup |
| Vector index | HNSW (node + edge) — O(log n) ANN |
| Secondary index trên `props` | **Chưa có** — full scan |
| Multi-tenancy | **Chưa có** — graph toàn cục |
| Referential integrity | **Chưa có** — orphaned edge có thể xảy ra |
| Stable external ID | `external_id` trên `NodeInfo` — persisted trong `nodes.json` |
| Edge ID | `edge_id: u64` monotonic — stable trong delta phase |
| ID phân tán | Instance ID partitioning, tối đa 256 instances |
| Merge remap | Đã fix — delta edge from/to được remap về CSR index đúng |
