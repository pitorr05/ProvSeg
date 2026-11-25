# v3-sota: baseline 2024–2026 trên cùng tài liệu test

**Nhánh:** `v3-sota`, tạo từ `v2-core` sau khi PASS. **Lệnh server:** `bash scripts/run_unattended.sh v3`. **Ước tính:** 2–4 giờ.

## Mục tiêu
Bảng so sánh với các phương pháp mới nhất cùng chủ đề. Thiếu bảng này, reviewer ICLR sẽ reject ngay.
- **Đã có sẵn** (`run_baselines.sh`): Seq-RoBERTa, Sent-RoBERTa (code SeqXGPT), Fast-DetectGPT, Binoculars (hàm chấm điểm chính thức, chạy theo câu).
- **Cần viết:** T1 và T2.

## Task
### T1: `scripts/baseline_markov.py`, Markov-informed calibration (ICLR'26)
- **Nguồn:** arXiv 2602.08031, repo `github.com/tmlr-group/MRF_Calibration`. Clone vào `../refs/MRF_Calibration`, ghi commit. **Ưu tiên import hàm gốc của repo**; chỉ tự viết lại khi không import được, và phải ghi rõ trong docstring.
- **Công thức** (đã đọc từ paper):
  - MRF cặp trên nhãn token y ∈ {0 người, 1 máy}.
  - Unary: Ψ_U = −log p(s_t), trong đó p là xác suất từ điểm detector.
  - Pairwise: Ψ_P = β(j)·w·(2·1[y_i ≠ y_j] − 1) với hàng xóm t±1.
  - Trọng số vị trí: β(t) = sigmoid(t − t0).
  - Mean-field: Q ← softmax(log H − A^corr Q (W ⊙ [[−1,1],[1,−1]])), bắt đầu từ Q⁰ = H. Chỉ học W (2×2) bằng CE.
- **Áp dụng cho bài của mình:**
  - Điểm token = cột `fdg_z` (Fast-DetectGPT) và `xent` (Binoculars) trong `data/feats_small/<ds>.npz`, tức đúng điểm S1 đang dùng.
  - p(s): logistic fit trên các từ của split train.
  - W và t0 học/chọn trên train/dev.
  - Nhãn mỗi từ = argmax Q_t; prob = Q_t.
  - Báo cả 2 biến thể (+fdg, +bino).
- **Đầu ra:** `../runs/baselines/markov_<score>_<ds>/pred_<ds>.pkl` với meta `step=baseline`, `arch=markov_<score>`.
- **Test:** chuỗi tổng hợp có nhiễu đơn lẻ thì sau MRF, số đoạn (fragmentation) phải giảm.

### T2: `scripts/baseline_wcp.py`, change-point WCP (arXiv 2605.03723)
- **Nguồn:** repo `github.com/Mamba413/DetectLLMSegmentation` (Python hoặc R; dùng code gốc nếu là Python).
- **Thuật toán:**
  - Điểm theo câu = trung bình `fdg_z` của các từ trong câu. Câu tách bằng `provseg.eval.metrics.sentence_spans`.
  - CUSUM có trọng số w_i = 1/σ̂_i² (σ̂ ước lượng theo độ dài câu như paper), thuật toán NOT với M khoảng ngẫu nhiên; ngưỡng r chọn trên dev.
  - Sau đó k-means k = 2 trên trung bình đoạn: cụm có trung bình cao hơn là AI.
- **Đầu ra:** như T1, với `arch=wcp`.
- **Test:** chuỗi có 1 điểm đổi rõ thì tìm đúng vị trí, sai lệch ≤ 1 câu.

### T3: thêm nhãn vào `eval_ladder.py`
Thêm `markov_fdg`, `markov_bino`, `wcp` vào PRETTY, kèm venue.

### T4: `significance.py --target S4` trên 1 seed
Chỉ để tham khảo; claim chính thức để v4 làm.

## Cổng
S4 ≥ baseline mạnh nhất (`BEST`) trên ≥ 2/3 benchmark.

## Nếu FAIL
- **Nếu Markov'26 thắng:** thử dùng chính lớp MRF có học làm Boundary Curator, tức thêm bước S2c.
  - Đóng góp lúc này là kết hợp có giám sát cộng bảo đảm cộng giải thích. Ghi DECISIONS.
- **Nếu WCP thắng ở WindowDiff nhưng thua ở F1:** báo cả hai metric. Đây là trade-off hợp lệ và nên nêu trong paper.
