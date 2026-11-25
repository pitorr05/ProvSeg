# Các phiên bản ProvSeg (nâng dần): mỗi phiên bản là một nhánh git, một prompt và một script chạy treo

```
main ── v1-baseline ── v2-core ── v3-sota ── v4-main ── v5-robust ── v6-paper
          (S0,S1)     (S2,S2b,S4)  (baseline    (3 seed, 5     (chéo miền,    (bảng, hình,
                                    2024-26)     benchmark,     OOD, giải      RESULTS_INDEX)
                                                 bootstrap)     thích, R-C)
```

## Quy trình cho mỗi phiên bản vN
1. **Trên máy có AI (không cần GPU):**
   - `git checkout vN-...` (nhánh đã có sẵn trên GitHub).
   - Mở phiên trợ lý AI và dán `versions/PROMPT_ITERATE.md` kèm tên phiên bản, ví dụ "làm v3".
   - Trợ lý đọc `versions/COMMON_RULES.md` và `versions/vN/PROMPT.md`, viết code còn thiếu, kiểm thử trên CPU, commit và push lên nhánh vN.
2. **Trên server GPU (chạy treo):**
   ```bash
   cd /data/ProvSeg && git fetch && git checkout vN-... && git pull
   tmux new -s vN        # hoặc dùng nohup
   bash scripts/run_unattended.sh vN
   ```
   Script tự chờ GPU trống ≥ 12GB ổn định 5 phút, chạy hết, verify, rồi ghi `../results/STATUS_vN.md`.
3. **Kiểm tra:** `cat ../results/STATUS_vN.md`.
   - **PASS:** merge vào main và tag `vN-pass`, rồi sang phiên bản tiếp theo.
   - **FAIL:** dán STATUS cùng 50 dòng cuối của log cho trợ lý, kèm `PROMPT_ITERATE.md`. Trợ lý sửa trên nhánh vN, rồi chạy lại. Các run đã xong sẽ được **tự bỏ qua** nhờ cơ chế resume.

## Trạng thái hiện tại (đã viết xong code của mọi phiên bản)
| Phiên bản | Code | Nhánh |
|---|---|---|
| v1 | đủ | `v1-baseline` |
| v2 | đủ (fp32 fix, Boundary Curator, chẩn đoán) | `v2-core` |
| v3 | đủ: `baseline_markov.py` (ICLR'26, theo code chính thức), `baseline_wcp.py` (2026) | `v3-sota` |
| v4 | đủ: loader CoAuthor (replay keystroke), tự tải dữ liệu | `v4-main` |
| v5 | đủ: `explain_deletion.py`, `plot_risk_coverage.py`, `parse_cost.py` | `v5-robust` |
| v6 | đủ: `make_paper_assets.py` | `v6-paper` (chứa toàn bộ code) |

Mỗi nhánh chứa code của nhánh trước nó, nên `v6-paper` có code của tất cả.

## Chạy tất cả bằng một lệnh (khuyến nghị khi có server)
```bash
cd /data/ProvSeg && source /data/venv/bin/activate && git fetch && git checkout v6-paper && git pull
tmux new -s provseg
bash scripts/run_all_versions.sh          # Ctrl+B rồi D để thoát ra; tmux attach -t provseg để xem lại
cat ../results/STATUS_ALL.md              # sáng hôm sau: PASS / GATE FAIL / CRASH của từng phiên bản
```
- **GATE FAIL:** cổng go/no-go không đạt. Vẫn chạy tiếp các phiên bản sau để gom đủ số liệu trong tháng có GPU.
- **CRASH:** lỗi code. Chuỗi dừng lại. Chạy lại lệnh trên thì các run đã xong được bỏ qua.
- Chỉ chạy một phần: `VERSIONS="v3 v4" bash scripts/run_all_versions.sh`.
