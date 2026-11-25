# v6-paper: đóng gói kết quả cho bản thảo ICLR

**Nhánh:** `v6-paper`, tạo từ `v5-robust`. **Lệnh:** `bash versions/v6/run.sh` (chỉ cần CPU).

## Mục tiêu
Mọi bảng và hình trong paper được tạo lại bằng **một lệnh** từ các file kết quả đã lưu. Sau v6 có thể trả GPU.

## Task
### T1: `scripts/make_paper_assets.py`
Tạo vào `paper/`:
- `tables/main.tex`: lấy từ `../results/significance/table_main.md` và `significance.csv`. Dùng booktabs, mean ± std, **in đậm** ô tốt nhất, dấu `*` khi CI > 0.
- `tables/ladder.tex`: ablation S0 → S1 → S2 → S2b → S4 (theo vai trò).
- `tables/robust.tex`: ma trận chéo miền, PASTED-OOD, chi phí.
- `tables/selective.tex`: Cov/Risk @0.05, @0.10, so với max-prob.
- **Hình** (copy hoặc vẽ lại theo cùng style):
  - figM1, figM2;
  - figF1 (failure modes);
  - risk-coverage;
  - deletion test;
  - 1 ví dụ giải thích theo từ (lấy từ `cases_*.html`, vẽ lại bằng matplotlib).
- `RESULTS_INDEX.md`: mỗi bảng/hình ghi rõ lệnh tạo ra nó, file nguồn và commit.

### T2: cập nhật `paper/main.tex`
- Các mục Experiments, Ablation, Robustness, Limitations: chỉ dùng số có trong `tables/`.
- Phần Introduction theo câu chuyện trong `PROMPT_ICLR.md`: phát hiện → 3 lỗi → 3 vai trò → bảo đảm và giải thích. Nêu khác biệt với Markov'26, WCP'26, Teja'25, RACE'26.

### T3: checklist reviewer
Tự chấm theo `PROMPT_ICLR.md` mục 3 và `SKILL_ICLR.md` mục 5. Ghi kết quả vào DECISIONS.

## Cổng
- Chạy `versions/v6/run.sh` trên máy sạch tạo lại đúng mọi bảng.
- Không có con số nào trong `main.tex` mà không có trong `tables/` (script tự kiểm bằng grep số).
