# PROMPT_ITERATE: prompt dùng mỗi lần làm hoặc sửa một phiên bản (dán nguyên khối, thay <vN>)

```text
Tôi đang làm paper A*/Q1 nộp ICLR 2028 về phát hiện nguồn gốc (Người/AI/Unknown) theo từng từ: ProvSeg.
Phiên bản cần làm: <vN>   (ví dụ: v3)
Kết quả mới nhất từ server (nếu có): <dán ../results/STATUS_<vN>.md + 50 dòng cuối log + bảng liên quan>

QUY TRÌNH BẮT BUỘC, theo đúng thứ tự, không bỏ bước:
0. ĐỌC: versions/COMMON_RULES.md, versions/<vN>/PROMPT.md, README.md, PROMPT_ICLR.md (mục 1: bản đồ SOTA),
   SKILL_ICLR.md, ../results/DECISIONS.md (nếu có). Chạy `git status`, `git log --oneline -5`,
   `git branch -a`; đảm bảo đang ở nhánh <vN>-* (nếu chưa có thì tạo từ nhánh phiên bản trước đã PASS).
1. KIỂM TRA HIỆN TRẠNG: liệt kê task Tk của <vN> nào đã xong/chưa xong (dựa trên code thật, không đoán).
   Nếu có kết quả server: đọc STATUS/VERIFY, xác định nguyên nhân FAIL (đọc code + log), nêu bằng chứng.
2. RESEARCH (chỉ khi task cần): với baseline/kỹ thuật mới, đọc paper + repo chính thức (link trong PROMPT_ICLR.md);
   ghi lại công thức/siêu tham số chính xác sẽ dùng và chỗ nào mình phải chuyển đổi (vd. doc->word).
   Không dùng thông tin không kiểm chứng; ghi rõ nguồn.
3. LÀM: viết/sửa code cho các task còn thiếu, theo đặc tả trong versions/<vN>/PROMPT.md. Mỗi file mới phải có
   docstring nêu nguồn gốc (paper/repo/commit) và mọi điểm khác với bản gốc.
4. VERIFY CỤC BỘ (bắt buộc trước khi push):
   a) python -m pytest -q tests: tất cả pass; thêm test cho mỗi hàm mới (ít nhất 1 test đúng/sai rõ ràng).
   b) Chạy thử CPU: cd ProvSeg && LIMIT=5 SETS=semeval8c bash versions/<vN>/run.sh (hoặc phần tương ứng):
      chạy hết, không NaN, skipped_batches 0, verify_run.py không báo lỗi sanity (cổng có thể FAIL vì 5 docs).
   c) Tự review diff: tìm bug, rò rỉ test vào train/chọn mô hình, so sánh không công bằng.
5. GIT: commit trên nhánh <vN>-* (tác giả "Pham Quang Toan <phamtoan23022005@gmail.com>", KHÔNG Co-Authored-By,
   không nhắc AI), push nhánh. KHÔNG merge vào main nếu STATUS chưa PASS.
6. BÁO CÁO cho tôi (tiếng Việt):
   - việc đã làm + file đổi + kết quả verify cục bộ (dán output ngắn);
   - đúng 1 lệnh server:  cd /data/ProvSeg && git fetch && git checkout <nhánh> && git pull && bash scripts/run_unattended.sh <vN>
   - thời gian/VRAM ước tính, cổng go/no-go sẽ được kiểm tra;
   - nếu có số thật: bảng + file nguồn + quyết định go/no-go + ghi vào ../results/DECISIONS.md;
   - cập nhật versions/<vN+1>/PROMPT.md nếu kết quả làm thay đổi kế hoạch (vd. bỏ một thành phần không giúp).
CẤM: bịa số; claim "vượt" khi chưa có ≥3 seed + CI bootstrap > 0; chỉnh siêu tham số theo test; xoá kết quả cũ
     mà không ghi lý do vào DECISIONS.md.
```
