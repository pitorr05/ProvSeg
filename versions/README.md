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

## Trạng thái hiện tại
| Phiên bản | Code | Chạy được ngay? |
|---|---|---|
| v1 | đủ | có (đã chạy một lần ngày 15/11 trên main) |
| v2 | đủ | **có**, nên chạy tiếp theo |
| v3 | thiếu `baseline_markov.py`, `baseline_wcp.py` | cần làm T1–T2 trước |
| v4 | thiếu loader CoAuthor | cần làm T1 trước (v3 phải xong) |
| v5 | thiếu `explain_deletion.py`, `plot_risk_coverage.py` | cần làm T2–T3 trước |
| v6 | thiếu `make_paper_assets.py` | cần làm T1 trước |

Script `run.sh` của mỗi phiên bản **dừng ngay** và in rõ task còn thiếu (`MISSING ... -> implement task Tk`), nên không tốn GPU vô ích.
