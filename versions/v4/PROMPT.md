# v4-main: bảng chính của paper (3 seed, 5 benchmark, kiểm định bootstrap)

**Nhánh:** `v4-main`, tạo từ `v3-sota` sau khi PASS. **Lệnh server:** `bash scripts/run_unattended.sh v4`.
**Ước tính:** 10–16 giờ, phần lớn là PASTED. Có thể chia nhỏ bằng `SETS=seqxgpt,semeval8c,beemo`, rồi `SETS=pasted,coauthor`. Nhờ cơ chế resume, run đã xong sẽ được bỏ qua.

## Mục tiêu
Bảng chính có mean ± std trên 3 seed (13, 21, 42), và CI 95% bootstrap của hiệu số giữa S4 và từng baseline.

## Task
### T1: `provseg/data/loaders/coauthor.py`, benchmark co-writing **thật**
- **Nguồn:** CoAuthor (Lee et al., CHI 2022). Dữ liệu tại `https://coauthor.stanford.edu`, gồm các session keystroke log (.jsonl) với gợi ý GPT-3.
- **Tác giả từng ký tự:**
  - Replay event `text-insert` / `text-delete` / `suggestion-select`.
  - Ký tự do người gõ là H; ký tự từ gợi ý được chấp nhận là A.
  - Văn bản cuối cùng được gán nhãn theo từ bằng đa số ký tự, dùng `char_labels_to_words` trong `provseg/data/schema.py`.
- **Split:** theo session id, dùng `assign_split`.
- **Đăng ký:**
  - Thêm vào `scripts/build_data.py` (`--only coauthor`).
  - Thêm bước tải dữ liệu vào `setup_server.sh` (biến `PROFILE`).
- **Test:** một session giả lập nhỏ, kiểm tra nhãn từ đúng.
- WCP (2026) cũng dùng CoAuthor, nên kết quả có thể so trực tiếp với số WindowDiff trong paper của họ (ghi rõ khác biệt về split).

### T2: đảm bảo mọi script tôn trọng `SEEDS`
`run_seqxgpt.sh`, `run_provseg.sh`, `run_baselines.sh` đã có. Markov/WCP là phương pháp tất định, chạy 1 lần.

### T3: `significance.py`
Kiểm tra lại cách bootstrap theo tài liệu. Xuất `table_main.md`.

## Cổng
CI 95% của (S4 − BEST) > 0 trên ≥ 3/5 benchmark. Ghi chi tiết từng benchmark vào DECISIONS.

## Nếu FAIL
- Báo trung thực: "ngang SOTA về F1". Đóng góp chuyển trọng tâm sang Cov@α/Risk (Auditor) và giải thích; v5 phải chứng minh hai phần này thật mạnh.
- **Không** thêm seed hay đổi siêu tham số sau khi đã thấy test.
