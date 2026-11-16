# SKILL_ICLR: các paper ICLR được accept làm như thế nào, và áp dụng cho ProvSeg

> Mục tiêu: nộp **ICLR 2028**, hạn dự kiến khoảng tháng 9–10/2027. GPU chỉ có trong khoảng 1 tháng nữa (H100 dùng chung), nên **toàn bộ thực nghiệm phải chạy xong trong tháng này**. Viết bài có thể làm sau.
> Venue của các paper dưới đây đã được kiểm tra qua trang ICLR, proceedings hoặc OpenReview vào tháng 11/2026. Link nằm ở cuối file.

---

## 1. Mười paper ICLR gần đây: họ làm gì để được accept

| # | Paper | ICLR | Đóng góp | Kiểu "câu chuyện" |
|---|---|---|---|---|
| 1 | **ACE**: Agentic Context Engineering | 2026 | Xây trên Dynamic Cheatsheet: gọi tên 2 lỗi (*brevity bias*, *context collapse*), tách thành 3 vai trò Generator → Reflector → Curator, cập nhật dạng delta, đặt tên "playbook" | **Gọi tên lỗi → tách vai trò → đặt tên mới**. +10.6% trên agent, +8.6% trên finance |
| 2 | **EditLens** | 2026 | Đổi bài toán: không còn "AI hay người" mà "AI sửa bao nhiêu" (liên tục). Nhãn tạo từ độ đo tương đồng | **Đặt lại bài toán** cho đúng thực tế; có dataset và model |
| 3 | **Beyond Raw Detection Scores** (Markov-informed calibration) | 2026 | 2 quan sát về điểm theo token: *Neighbor Similarity* và *Initial Instability*. Dùng MRF với mean-field làm module gọn, gắn vào mọi detector đều tăng điểm | **Quan sát → module gắn thêm → tăng cho mọi baseline**. Là đối thủ gần nhất của mình |
| 4 | **SAFER** | 2026 | Kiểm soát rủi ro (conformal) cho đầu ra LLM, có bảo đảm hữu hạn mẫu | **Bảo đảm có chứng minh**; mẫu cho phần S3 |
| 5 | **Lastde** | 2025 | Xem chuỗi xác suất token như chuỗi thời gian; kết hợp thống kê cục bộ và toàn cục; không cần train | **Góc nhìn mới cho tín hiệu cũ** |
| 6 | **Glimpse** | 2025 | Cho phương pháp white-box chạy được trên model đóng (API) | **Mở rộng phạm vi áp dụng** của method cũ |
| 7 | **Humanizing the Machine** | 2025 | Tấn công làm các detector zero-shot thất bại | **Robustness / phản biện**: reviewer luôn hỏi |
| 8 | **Black-box watermark detection** | 2025 | Kiểm định thống kê chặt chẽ đầu tiên cho bài toán | **Rigor**: ICLR đánh giá cao kiểm định đúng |
| 9 | **Fast-DetectGPT** | 2024 | Đo curvature xác suất có điều kiện (AI ≈ 3, người ≈ 0); nhanh hơn DetectGPT 340 lần | **1 quan sát → method đơn giản và nhanh** |
| 10 | **Raidar** | 2024 | LLM sửa văn bản người nhiều hơn văn bản AI, nên dùng khoảng cách chỉnh sửa; +29 F1 | **Insight trực giác** → method chỉ cần black-box |

---

## 2. Công thức chung (rút ra từ 10 paper)

1. **Một quan sát hoặc phát hiện bất ngờ, có hình ở trang 1–2.**
   - Ví dụ: Fast-DetectGPT (curvature), Markov (2 quan sát), Raidar (LLM sửa người nhiều hơn).
   - Quan sát phải **đo được trên dữ liệu thật**, không phải cảm tính.
2. **Gọi tên lỗi của phương pháp cũ** bằng thuật ngữ dễ nhớ, kiểu *brevity bias* hay *context collapse* trong ACE.
3. **Method suy ra trực tiếp từ quan sát.**
   - Mỗi thành phần sửa đúng một lỗi đã gọi tên.
   - Method đơn giản, gọn, gắn được vào thứ có sẵn.
4. **Thắng rõ trên nhiều benchmark**, kèm baseline mới nhất (1–2 năm gần đây) và giao thức công bằng.
5. **Ablation từng thành phần**, sao cho mỗi lỗi đã gọi tên tương ứng với một dòng trong bảng.
6. **Robustness:** đổi LLM sinh văn bản (cross-LLM), paraphrase/tấn công, đổi domain.
7. **Chi phí tính toán:** thời gian và bộ nhớ. Cả Markov lẫn Fast-DetectGPT đều nhấn mạnh "overhead không đáng kể".
8. **Lý thuyết hoặc bảo đảm ngắn** nếu có (SAFER, watermark): ICLR coi trọng tính đúng.

**Tiêu chí reviewer ICLR** (Reviewer Guide 2026, blog ICLR):
- **Correctness:** các claim có được đỡ bằng bằng chứng thuyết phục không.
- **Significance:** phát hiện có *mới, bất ngờ, quan trọng* không.
- Điểm chấm: soundness, presentation, contribution. Có vòng rebuttal, nên chuẩn bị sẵn thí nghiệm bổ sung.

---

## 3. Mẫu ACE, viết thành các bước làm được

ACE không phát minh từ số 0. Họ đi 5 bước:
1. Chọn **một method gốc mạnh và nổi tiếng** (Dynamic Cheatsheet).
2. **Chạy nó, tìm chỗ hỏng, đặt tên cho chỗ hỏng.**
3. **Tách method thành các vai trò chuyên biệt**, mỗi vai trò sửa một chỗ hỏng.
4. **Tổ chức kỹ phần trung tâm** (cheatsheet được tổ chức lại thành "playbook" với cập nhật delta) và **đặt tên mới** cho nó.
5. Thắng trên benchmark khó nhất; ablation từng vai trò.

### Áp dụng cho ProvSeg (gốc là SeqXGPT, EMNLP'23)
| Lỗi của SeqXGPT (đặt tên) | Đo bằng | Vai trò trong ProvSeg | Bước |
|---|---|---|---|
| **Score myopia**: mỗi từ chấm riêng bằng log-prob thô, nhiễu | `fragmentation` (segment dự đoán / segment thật) | **Evidence Reader**: bằng chứng tương phản cộng ngữ cảnh DeBERTa | S1, S2 |
| **Boundary blur**: lỗi dồn quanh điểm chuyển người↔AI | `err_dist_*` (lỗi theo khoảng cách tới ranh giới) | **Boundary Curator**: head "segment bắt đầu ở đây", hợp nhất vào giải mã BMES | S2b |
| **Forced commitment**: luôn phải trả lời H/AI | `share_errors_in_low20_conf` | **Auditor**: Unknown có bảo đảm (Learn-then-Test) cộng giải thích Shapley | S3, S4 |

Các con số trên được đo bằng `scripts/diagnose.py` và ra hình `figF1_failure_modes.pdf`. Đây là **motivation lấy từ kết quả thật**.

**Khác biệt với đối thủ gần nhất (Markov calibration, ICLR'26):**
- Họ làm mượt điểm token **không giám sát** để ra điểm cấp **văn bản**.
- ProvSeg học ranh giới **có giám sát**, ra nhãn **theo từng từ**, cộng Unknown có bảo đảm và giải thích.
- Bắt buộc phải chạy baseline của họ (repo chính thức `tmlr-group/MRF_Calibration`).

---

## 4. Góc nhìn rút ra từ kết quả đã có (lượt chạy 15/11/2026, model nhỏ)

| | Beemo (AI viết → người sửa) | SemEval-8C (người → AI viết tiếp) | SeqXGPT-Bench |
|---|---|---|---|
| S0 SeqXGPT | 70.0 | 87.3 (MAE 28.5) | 89.8 |
| S1 + bằng chứng tương phản | 71.9 | 88.0 (MAE 27.0) | 89.1 |

1. **Văn bản AI bị người sửa (Beemo) khó hơn 17–20 điểm** so với kiểu người viết rồi AI viết tiếp. Đây là trục câu chuyện: *các detector hiện tại được thiết kế cho kiểu "AI viết tiếp", nhưng thực tế người và AI viết đan xen nhau*.
2. **Bằng chứng tương phản giúp đúng ở chỗ khó:** Beemo +1.9, ranh giới SemEval tốt hơn 1.5 từ. Không giúp trên SeqXGPT-Bench, nơi log-prob thô đã đủ.
3. **Ở mức α = 5%, Unknown không trả lời được từ nào** với calibrator cũ.
   - Lý do: calibrator kiểm tra lần lượt từ ngưỡng cao xuống thấp. Ở ngưỡng đầu tiên chỉ có vài tài liệu được trả lời, kiểm định quá yếu nên thất bại, và nó dừng luôn ở đó.
   - Đã đổi sang Bonferroni trên lưới ngưỡng cố định, và báo cả α = 0.05 lẫn α = 0.10.
4. **S2/S4 trước đây hỏng do NaN** khi DeBERTa chạy bf16. Đã sửa: encoder luôn chạy fp32 và có chặn gradient NaN.

---

## 5. Checklist trước khi nộp
- [ ] Baseline 2024–2026: SeqXGPT, Seq-RoBERTa, Fast-DetectGPT, Binoculars, **Markov calibration (ICLR'26)**; thêm HACo-Det/FAID nếu có code.
- [ ] ≥ 3 seed, báo mean ± std, cùng tập test cho mọi phương pháp.
- [ ] Ablation: S0 → S1 → S2 → S2b → S3 → S4.
- [ ] Robustness: train trên một benchmark, test trên benchmark khác; paraphrase.
- [ ] Chi phí: thời gian và VRAM mỗi bước (script đã in `peak VRAM`).
- [ ] Hình trang 1: failure modes (`diagnose.py`) cộng một ví dụ giải thích theo từ (`cases_*.html`).
- [ ] Mô tả rõ nguồn gốc dữ liệu và điểm số (xem phần mô tả M1/M2).
- [ ] Code công khai, giấy phép dữ liệu (EditLens CC BY-NC-SA).

## 6. Kế hoạch dùng GPU trong 1 tháng (ưu tiên số liệu trước)
1. **Tuần 1:**
   - S2, S2b, S4 sau khi đã sửa NaN.
   - Chạy `diagnose.py`.
   - Nếu S2b/S4 thắng S0 rõ thì chốt thiết kế.
2. **Tuần 2:**
   - 3 seed cho S0, S1, S2, S2b, S4.
   - `run_baselines.sh`.
   - Baseline Markov (ICLR'26).
3. **Tuần 3:** cross-domain (train benchmark A, test benchmark B), thêm PASTED và APT.
4. **Tuần 4:** chạy lại những gì thiếu, gom hết `results/`. Sau đó có thể viết bài mà không cần GPU.

---

### Nguồn
- ACE: https://iclr.cc/virtual/2026/10021286 · https://arxiv.org/abs/2510.04618
- EditLens: https://iclr.cc/virtual/2026/poster/10008153
- Markov-informed calibration: https://www.iclr.cc/virtual/2026/poster/10009986 · https://arxiv.org/abs/2602.08031
- SAFER: https://mlanthology.org/iclr/2026/wang2026iclr-safer/
- Lastde: https://iclr.cc/virtual/2025/poster/27889
- Glimpse: https://proceedings.iclr.cc/paper_files/paper/2025/file/806e8e4ea77000a1486eeddeb1b90a6e-Paper-Conference.pdf
- Humanizing the Machine: https://proceedings.iclr.cc/paper_files/paper/2025/file/ab1ee157f7804a13f980414b644a9460-Paper-Conference.pdf
- Black-box watermark detection: https://proceedings.iclr.cc/paper_files/paper/2025/file/c0f7ee1901fef1da4dae2b88dfd43195-Paper-Conference.pdf
- Fast-DetectGPT: https://iclr.cc/virtual/2024/poster/19201
- Raidar: https://proceedings.iclr.cc/paper_files/paper/2024/file/1888b9df34b0d9eea6e009a1fdd55c4f-Paper-Conference.pdf
- ICLR 2026 Reviewer Guide: https://iclr.cc/Conferences/2026/ReviewerGuide · ICLR blog: https://blog.iclr.cc/?p=1150
