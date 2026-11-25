# PROMPT_ICLR: prompt điều phối dự án ProvSeg tới paper A*/Q1 (ICLR 2028)

> Cách dùng: mở một phiên mới với trợ lý AI trong thư mục repo `ProvSeg/`, dán **toàn bộ khối PROMPT** ở mục 2. Đính kèm bảng kết quả mới nhất (`../results/compare/table_ladder.md`, `../results/diagnose/failure_modes.csv`).
> Mục 1 là bản đồ SOTA 2025–2026 (đã kiểm tra venue và đọc paper, tháng 11/2026). Prompt sẽ tham chiếu tới mục này.

**Lưu ý trung thực:** không prompt nào *bảo đảm* paper được accept hay *bảo đảm* thắng. Thứ prompt này bảo đảm được là:
1. Mọi claim trong paper đều có số liệu thật chống lưng (có kiểm định thống kê).
2. Mỗi bước có cổng **go/no-go** rõ ràng, nên không tốn GPU vào hướng thua.
3. Có sẵn đường lùi: một câu chuyện vẫn đứng được nếu một thành phần không thắng.

---

## 1. Bản đồ SOTA cùng chủ đề (2025–2026) và chỗ trống cho ProvSeg

| Paper (venue) | Mức nhãn | Kiến trúc | Dữ liệu | Code | Quan hệ với ProvSeg |
|---|---|---|---|---|---|
| **SeqXGPT** (EMNLP'23) | từ / câu | log-prob của 4 LLM → CNN → Transformer → BMES-Viterbi | SeqXGPT-Bench | có | **Gốc (S0)**, đã tái lập: 87–90 F1 |
| **Markov-informed calibration** (ICLR'26) | token, dùng để ra quyết định cấp văn bản | MRF cặp trên nhãn token; unary = −log p(điểm detector); pairwise thưởng hàng xóm cùng nhãn, có trọng số vị trí β(t) dạng sigmoid; mean-field lặp, chỉ học ma trận 2×2 | Essay, Reuters, DetectRL, TruthfulQA; cross-LLM (888 setting), paraphrase | `tmlr-group/MRF_Calibration` | **Đối thủ trực tiếp** của Boundary Curator. Khác biệt: họ làm mượt không giám sát và ra quyết định cấp văn bản. Có thể dùng như một **lớp thay thế** trong ablation |
| **WCP: change-point segmentation** (arXiv 07/2026) | câu | Không train: CUSUM có trọng số (w = 1/σ̂², tức inverse-variance) trên chuỗi điểm detector (AdaDetectGPT / Fast-DetectGPT), thuật toán NOT, rồi k-means 2 cụm; có chứng minh minimax-optimal | WikiQA/News/Story (bán tổng hợp), **CoAuthor (thật)**; metric **WindowDiff** | `Mamba413/DetectLLMSegmentation` | **Baseline phân đoạn bắt buộc**; nên dùng metric WindowDiff của họ |
| **RACE** (ACL'26) | văn bản, 4 lớp (người / LLM / LLM-polish người / người-hoá LLM) | RoBERTa-base (chỉ fine-tune lớp cuối) → cây RST → RGCN 2 lớp → SupCon + CE | HART | `ICTMCG/RACE` | Khác mức nhãn. Dùng để **trích dẫn**: "creator vs editor" chỉ được làm ở cấp văn bản, còn ProvSeg làm theo từng từ |
| **FAID** (EACL'26) | văn bản, 3 lớp + họ LLM | encoder + contrastive đa mức + multi-task | FAIDSet (~83k) | `mbzuai-nlp/FAID` | Trích dẫn; có thể dùng FAIDSet làm benchmark ngoài miền |
| **Teja et al.** (IJCNLP-Findings'25) | token, gộp lên câu | Transformer + **CRF + boundary-aware loss** + regularization nhận biết chuỗi | hỗn hợp | — | **Rủi ro trùng ý với S2b** (Boundary Curator). Phải nêu rõ khác biệt, hoặc dùng làm baseline |
| **HACo-Det** (ACL'25) | từ → câu | chuyển 7 detector cấp văn bản xuống mức từ; F1 tốt nhất 0.462 | HACo-Det (paraphrase nhiều lượt) | dữ liệu? | Benchmark co-authoring khó; motivation "bài toán còn mở" |
| **EditLens** (ICLR'26) | văn bản, liên tục | regression mức độ AI sửa | EditLens | có | M1 dùng điểm có sẵn; trích dẫn cho cách đặt bài toán |
| DAMASHA, Mixture of Detectors (arXiv'25) | token | DeBERTa-BiGRU-CRF; token-F1 ≈ 0.98 nhưng span-F1 ≈ 0.45 | hỗn hợp, có tấn công | ? | Bằng chứng rằng **định vị span vẫn khó** dù token-F1 cao |

**Chỗ trống mà chưa paper nào ở trên làm** (đây là đóng góp hợp lệ):
1. **Nhãn theo từ kèm "Unknown" có bảo đảm hữu hạn mẫu** (lỗi ≤ α với xác suất ≥ 1−δ). Paper Markov và WCP đều không có.
2. **Giải thích chính xác theo nguồn bằng chứng cho từng từ** (Shapley theo kênh, tổng đúng bằng margin).
3. **Kết hợp có giám sát cả 3 loại tín hiệu** (điểm zero-shot theo token, ngữ cảnh encoder, mô hình ranh giới) và so trực tiếp với từng loại riêng lẻ: MRF (ICLR'26), WCP (2026), CRF + boundary loss (2025).
4. **Phát hiện đo lường M1/M2:** điểm detector không đo được mức đóng góp (ρ ≈ 0.2 trên văn bản AI-sửa), và % AI phụ thuộc cách định nghĩa (lệch 48–62 điểm).

**Rủi ro đã biết:**
- Boundary Curator một mình **không đủ mới**, vì Teja'25 đã có CRF + boundary loss và Markov'26 đã có làm mượt hàng xóm.
- Vậy đóng góp phương pháp phải là **khung 3 vai trò** cộng **bảo đảm** cộng **giải thích**, không phải từng module riêng lẻ.

---

## 2. PROMPT (dán nguyên khối)

```text
VAI TRÒ
Bạn là đồng tác giả kỹ thuật kiêm reviewer ICLR khó tính. Mục tiêu: hoàn thành TẤT CẢ thực nghiệm GPU cho
paper ProvSeg trong ≤ 4 tuần (GPU H100 dùng chung, mỗi job ≤ ~12GB VRAM), để nộp ICLR 2028.
Viết bài có thể làm sau. Repo: ProvSeg/ (đọc README.md, SKILL_ICLR.md, PROMPT_ICLR.md mục 1 trước khi làm gì).

NGUYÊN TẮC BẤT BIẾN (vi phạm = dừng lại)
1. Không bịa số. Mọi con số trong bảng/văn bản phải truy được về một file trong ../results/.
2. Test set chỉ dùng để báo cáo cuối. Chọn mô hình/siêu tham số trên dev; calibrator trên cal.
3. Mỗi claim "tốt hơn" cần: ≥3 seed (mean±std) VÀ paired bootstrap 1000 lần trên tài liệu test,
   khoảng tin cậy 95% của hiệu số không chứa 0. Không đạt -> viết "ngang bằng", không viết "vượt".
4. Mọi phương pháp chạy trên CÙNG tài liệu test, cùng ngân sách LLM đặc trưng (model nhỏ), cùng split.
5. Baseline chạy bằng code/hàm chấm điểm chính thức; mọi chuyển đổi (vd. doc->word) ghi rõ.
6. Mỗi thay đổi code: có test, commit với tác giả "Pham Quang Toan <phamtoan23022005@gmail.com>",
   không thêm dòng Co-Authored-By hay nhắc tới AI.
7. Mỗi tuần kết thúc bằng một bảng go/no-go viết vào ../results/DECISIONS.md.

TRẠNG THÁI HIỆN TẠI (cập nhật từ bảng tôi đính kèm)
- S0 SeqXGPT (model gốc, LLM nhỏ): Beemo 70.0 / SemEval-8C 87.3 (MAE 28.5) / SeqXGPT-Bench 89.8 word-F1.
- S1 (+ bằng chứng tương phản): 71.9 / 88.0 (MAE 27.0) / 89.1.
- S2 (DeBERTa), S2b (Boundary Curator), S4 (evidence dropout + Shapley): đã sửa lỗi NaN (fp16 weights +
  AdamW), CHƯA có số thật -> cần kết quả lần chạy mới.
- Calibrator: Bonferroni-LTT, báo Cov/Risk ở α = 0.05 và 0.10.

TUẦN 1: CHỐT KIẾN TRÚC (go/no-go)
a) Chạy S2, S2b, S4 trên 3 benchmark (STEPS="S2 S2b S4 eval" bash run_provseg.sh) và scripts/diagnose.py.
b) Thêm 2 baseline SOTA 2026 vào run_baselines.sh:
   - Markov-informed calibration (ICLR'26, github tmlr-group/MRF_Calibration): áp lên chuỗi điểm token
     Fast-DetectGPT/Binoculars đã có trong data/feats_small; xuất nhãn theo từ = argmax Q_t.
   - WCP change-point (2026, github Mamba413/DetectLLMSegmentation): chạy trên điểm theo câu.
c) Thêm metric WindowDiff (k = nửa độ dài segment trung bình) và span-F1@IoU0.5 vào eval_ladder.py.
CỔNG GO:   S2 hoặc S2b vượt S0 ≥ +2 word-F1 trên ≥2/3 benchmark VÀ không thua Markov/WCP trên Beemo.
NẾU NO-GO: (i) thử DeBERTa-v3-large với gradient checkpointing nếu VRAM cho phép; (ii) thử thay head
           Boundary Curator bằng lớp MRF mean-field có học (khác Markov'26: học có giám sát, ra nhãn từ);
           (iii) nếu vẫn không thắng: chuyển trục paper sang "đo lường + bảo đảm" (xem ĐƯỜNG LÙI).

TUẦN 2: SỐ CHÍNH CỦA PAPER
- 3 seed cho S0, S1, S2, S2b, S4 + baseline có giám sát (Seq-RoBERTa, Sent-RoBERTa) + zero-shot
  (Fast-DetectGPT, Binoculars) + Markov'26 + WCP'26.
- Thêm benchmark co-authoring thật: CoAuthor (refs/coauthor-interface) và PASTED (đã có loader).
- Viết scripts/significance.py: paired bootstrap theo tài liệu, xuất CI cho mọi cặp (ProvSeg vs từng baseline).
CỔNG GO: ProvSeg (S4) có CI dương so với baseline mạnh nhất trên ≥3/5 benchmark.

TUẦN 3: ROBUSTNESS + HIỆU NĂNG + BẢO ĐẢM
- Cross-domain: train benchmark A, test benchmark B (ma trận 3x3).
- Tấn công: paraphrase (dùng PASTED-OOD), xoá/hoán vị câu nhẹ.
- Bảo đảm: vẽ risk-coverage; kiểm tra Risk@α thực tế ≤ α trên test ở mọi benchmark; so với
  ngưỡng "max-prob" thường (không bảo đảm) để cho thấy calibrator mới là thứ giữ được α.
- Chi phí: thời gian/doc và peak VRAM mỗi phương pháp.
- Giải thích: kiểm tra "deletion test" (xoá kênh có |phi| lớn nhất phải làm margin giảm nhiều hơn xoá
  kênh ngẫu nhiên), bảng tỷ trọng |phi| theo kênh trên từ AI-sửa vs AI-viết.

TUẦN 4: LẤP LỖ + ĐÓNG GÓI
- Chạy lại mọi ô trống, gom ../results/, viết RESULTS_INDEX.md (mỗi bảng/hình -> lệnh tạo ra nó).
- Xuất hình: figF1 (failure modes), figM1, figM2, risk-coverage, ví dụ giải thích theo từ.

CÂU CHUYỆN PAPER (chỉ viết phần nào có số đỡ)
Tiêu đề làm việc: "Who Wrote What Survived? Certified, Evidence-Grounded Word-Level Provenance for
Human-AI Co-Written Text".
1. Phát hiện (trang 1, hình): điểm detector không đo được đóng góp (M1); % AI phụ thuộc định nghĩa (M2);
   SeqXGPT hỏng theo 3 kiểu đo được (figF1): score myopia, boundary blur, forced commitment.
2. Method = 3 vai trò, mỗi vai trò sửa một lỗi (kiểu ACE ICLR'26): Evidence Reader -> Boundary Curator
   -> Auditor (LTT + Shapley). Nêu rõ khác Markov'26 (không giám sát, cấp văn bản), WCP'26 (không học,
   cấp câu), Teja'25 (không bảo đảm, không giải thích).
3. Lý thuyết ngắn: bảo đảm LTT hữu hạn mẫu theo tài liệu; tính chất efficiency của Shapley theo kênh.
4. Thực nghiệm: bảng chính (5 benchmark x 10 phương pháp, mean±std, đánh dấu CI), ablation theo vai trò,
   robustness, chi phí, risk-coverage, case study.
ĐƯỜNG LÙI (nếu không thắng F1 rõ): paper "đo lường + bảo đảm": đóng góp chính = M1/M2 + giao thức đánh giá
   + Auditor có bảo đảm áp lên MỌI detector (kể cả baseline), cho thấy Cov@α tăng mà Risk ≤ α.
   Đây vẫn là đóng góp ICLR hợp lệ (kiểu "Beyond Raw Detection Scores" là module gắn thêm).

ĐẦU RA MỖI LẦN BẠN TRẢ LỜI
(1) việc đã làm + file đã đổi, (2) lệnh cho server (có vòng chờ GPU: chờ free ≥ 12GB),
(3) bảng số mới nhất kèm đường dẫn file nguồn, (4) quyết định go/no-go và lý do, (5) rủi ro còn lại.
```

---

## 3. Checklist reviewer (tự chấm trước khi nộp)
- [ ] Claim chính có CI 95% không chứa 0 so với baseline mạnh nhất (Markov'26, WCP'26, Seq-RoBERTa/DeBERTa-CRF).
- [ ] Có ít nhất 1 benchmark co-authoring *thật* (CoAuthor), không chỉ dữ liệu ghép.
- [ ] Ablation chứng minh từng vai trò có đóng góp (bỏ từng vai trò thì điểm giảm, có CI).
- [ ] Bảo đảm: Risk thực tế ≤ α trên mọi benchmark; nêu rõ giả định (tài liệu exchangeable).
- [ ] Khác biệt với Teja'25 và Markov'26 được nêu rõ ngay ở phần Introduction.
- [ ] Nêu các giới hạn: LLM đặc trưng nhỏ, tiếng Anh, chỉ một vòng sửa.

## Nguồn
- SeqXGPT: https://aclanthology.org/2023.emnlp-main.73.pdf
- Markov-informed calibration (ICLR'26): https://arxiv.org/abs/2602.08031 · https://www.iclr.cc/virtual/2026/poster/10009986
- WCP change-point: https://arxiv.org/abs/2605.03723
- RACE (ACL'26): https://aclanthology.org/2026.acl-long.235/ · https://arxiv.org/abs/2604.04932
- FAID (EACL'26): https://aclanthology.org/2026.eacl-long.151
- Teja et al. (IJCNLP-Findings'25): https://aclanthology.org/2025.findings-ijcnlp.48/
- HACo-Det (ACL'25): https://aclanthology.org/2025.acl-long.1069
- EditLens (ICLR'26): https://iclr.cc/virtual/2026/poster/10008153
- DAMASHA: https://arxiv.org/abs/2512.04838 · Mixture of Detectors: https://arxiv.org/abs/2509.22147
- ACE (ICLR'26): https://arxiv.org/abs/2510.04618
