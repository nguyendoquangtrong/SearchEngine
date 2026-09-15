"""Build the Vietnamese ground-truth report used for the paper revision."""
import json
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "groundtruth" / "generated"
OUT = ROOT / "ground_truth.docx"


def shade(cell, fill):
    props = cell._tc.get_or_add_tcPr()
    node = OxmlElement("w:shd")
    node.set(qn("w:fill"), fill)
    props.append(node)


def set_cell_text(cell, text, bold=False):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(9)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, value in enumerate(headers):
        set_cell_text(table.rows[0].cells[i], value, True)
        shade(table.rows[0].cells[i], "D9EAF7")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell_text(cells[i], value)
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Cm(width)
    doc.add_paragraph()
    return table


def bullet(doc, text):
    doc.add_paragraph(text, style="List Bullet")


def main():
    all_metrics = json.loads((GEN / "metrics_ALL_KEPT.json").read_text(encoding="utf-8"))
    supported_metrics = json.loads((GEN / "metrics_SUPPORTED_ONLY.json").read_text(encoding="utf-8"))
    summary = json.loads((GEN / "groundtruth_summary_USER_REVIEWED.json").read_text(encoding="utf-8"))

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    normal.font.size = Pt(10.5)

    title = doc.add_heading("GROUND TRUTH VÀ ĐÁNH GIÁ HỆ THỐNG TÌM KIẾM PHIM", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = doc.add_paragraph("Tài liệu bổ sung cho bản sửa bài báo")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].italic = True

    doc.add_heading("1. Mục tiêu", level=1)
    doc.add_paragraph(
        "Tài liệu này mô tả việc xây dựng lại ground truth cho hệ thống Movie Search, "
        "các thay đổi được thực hiện sau phản biện, và kết quả đánh giá lại V1. Mục tiêu là "
        "thay thế tập truy vấn nhỏ, chưa có quy trình xác thực rõ ràng bằng một bộ đánh giá có "
        "nguồn gốc, nhãn, tiêu chí loại trừ, tách development/test và kết quả tái lập được."
    )

    doc.add_heading("2. Các vấn đề được reviewer nêu ra", level=1)
    add_table(doc,
              ["Vấn đề", "Điều chỉnh đã thực hiện"],
              [
                  ["Tập đánh giá nhỏ và chưa phản ánh truy vấn người dùng thực tế.",
                   "Sử dụng khảo sát 61 người dùng, mỗi người tạo 3 dạng truy vấn; tổng cộng 183 query."],
                  ["Ground truth chưa rõ cách gán và kiểm chứng.",
                   "Lưu query thô, query chuẩn hóa, target do người trả lời chỉ định, cờ QC, quyết định review và qrels theo từng query."],
                  ["Có query nhiễu, không đủ thông tin, hoặc target không được hỗ trợ.",
                   "Owner review từng query; 9 query bị loại, 29 query được giữ nhưng đánh dấu partly supported."],
                  ["Thiếu tách tập phát triển và tập kiểm tra; thiếu bất định thống kê.",
                   "Tách theo người dùng: 15 người/42 query dev và 45 người/132 query test; duplicate cluster không bị tách. Báo cáo bootstrap CI 95% theo cụm người dùng."],
                  ["Cần so sánh rõ ràng với các thành phần/baseline.",
                   "Đánh giá cùng một tập query cho BM25, SBERT, CLIP Text, CLIP Image và V1 PT2."],
              ], [5.6, 10.7])

    doc.add_heading("3. Quy trình xây dựng ground truth", level=1)
    numbered = [
        "Chuẩn hóa và lưu toàn bộ 183 query khảo sát cùng participant ID, nhóm prompt, target movie do người trả lời chọn và movie ID trong corpus.",
        "Kiểm tra dữ liệu corpus: 62 phim, 3.213 frame ảnh; các collection text có 38.202 mục. Target của khảo sát đều ánh xạ được vào movie ID của corpus.",
        "Chạy kiểm tra tự động để phát hiện query ngắn, title leakage, query không phản hồi, truy vấn trùng và trường hợp có tín hiệu yếu với transcript/caption.",
        "Thực hiện owner review. Query được phân thành supported, partly supported hoặc exclude. Partly supported vẫn có target hợp lý nhưng câu chữ/cảnh/quote không hoàn toàn chính xác.",
        "Xuất qrels movie-level dạng TREC: query_id, movie_id và relevance = 1. Các query exclude không có qrel và không tham gia tính metric.",
        "Tách dữ liệu theo participant với seed 4204. Việc giữ query trùng trong cùng split tránh rò rỉ thông tin giữa dev và test.",
    ]
    for item in numbered:
        doc.add_paragraph(item, style="List Number")

    doc.add_heading("4. Thống kê nhãn sau review", level=1)
    add_table(doc, ["Nhãn", "Số query", "Cách sử dụng"], [
        ["Supported", summary["supported"], "Dùng trong đánh giá chính và sensitivity analysis."],
        ["Partly supported", summary["partly_supported"], "Dùng trong đánh giá chính; báo cáo riêng sensitivity analysis khi loại nhóm này."],
        ["Exclude", len(summary["excluded"]), "Không đưa vào qrels hoặc metric."],
        ["Tổng", summary["input_queries"], "183 query từ khảo sát."],
        ["Qrels hợp lệ", summary["included_queries"], "174 query dùng để đánh giá."],
    ], [4.2, 2.4, 9.7])
    doc.add_paragraph("Các query bị loại và lý do:")
    add_table(doc, ["Query ID", "Lý do loại"], [
        ["P009_G3", "Mô tả cảnh người đàn ông lớn tuổi mặc áo vest đen hút xì gà không được xác nhận đáng tin cho target The Godfather."],
        ["P010_G1", "Quote không được xác minh bằng transcript/nguồn phim phù hợp; không đủ căn cứ dùng làm truy vấn quote có ground truth."],
        ["P027_G1", "Câu quote không phải lời thoại xác nhận được của Iron Man 3; có dấu hiệu là câu nói bị gán sai trên Internet."],
        ["P033_G1", "Không có bằng chứng đáng tin cho câu quote trong City of God; kết quả matcher cũ cũng trỏ tới đoạn không liên quan."],
        ["P041_G1", "“I can't remember.” là câu trả lời quá chung, không đủ tín hiệu để xác định nhu cầu tìm kiếm hoặc target phim."],
        ["P041_G2", "“Quiet yet powerful.” là nhận xét chủ quan, không mô tả plot, nhân vật, thoại hoặc cảnh để thực hiện retrieval."],
        ["P053_G1", "Chỉ chứa tên phim và tên nhân vật; title leakage, không phải truy vấn tìm kiếm tự nhiên."],
        ["P053_G2", "Chứa tên phim và từ khóa quá nghèo thông tin; title leakage, không đủ để đánh giá retrieval công bằng."],
        ["P053_G3", "Không mô tả nội dung/cảnh cần tìm, chứa title leakage và ngôn từ không phù hợp để dùng làm query đánh giá."],
    ], [3.2, 12.9])

    doc.add_heading("5. Thiết kế đánh giá", level=1)
    bullet(doc, "Đơn vị đánh giá: movie-level retrieval; một target movie do người dùng nêu được xem là relevant.")
    bullet(doc, "Hệ thống: BM25, SBERT, CLIP Text, CLIP Image và V1 PT2 (fusion).")
    bullet(doc, "Metrics: Success@1, Success@5 và MRR@5.")
    bullet(doc, "Test chính: 132 query từ 45 người dùng. Dev: 42 query từ 15 người dùng.")
    bullet(doc, "Khoảng tin cậy: bootstrap 10.000 lần, resampling theo participant để phản ánh sự phụ thuộc giữa 3 query của cùng người dùng.")
    bullet(doc, "Các run được export và đóng băng trước khi tính metric; không dùng test set để chỉnh trọng số fusion.")

    def metric_rows(report):
        return [[name, f"{val['success_at_1']:.4f}", f"{val['success_at_5']:.4f}", f"{val['mrr_at_5']:.4f}"]
                for name, val in report["test"]["systems"].items()]

    doc.add_heading("6. Kết quả trên test: tất cả query hợp lệ", level=1)
    doc.add_paragraph("Tập test gồm 132 query: supported và partly supported.")
    add_table(doc, ["Hệ thống", "Success@1", "Success@5", "MRR@5"], metric_rows(all_metrics), [4.7, 3.3, 3.3, 3.3])
    doc.add_paragraph(
        "V1 PT2 có điểm cao nhất: Success@1 = 0.5530, Success@5 = 0.7576 và MRR@5 = 0.6318. "
        "BM25 là baseline cạnh tranh nhất với MRR@5 = 0.5995."
    )

    doc.add_heading("7. Sensitivity analysis: chỉ query supported", level=1)
    doc.add_paragraph("Tập test còn 108 query khi loại 24 query partly supported.")
    add_table(doc, ["Hệ thống", "Success@1", "Success@5", "MRR@5"], metric_rows(supported_metrics), [4.7, 3.3, 3.3, 3.3])
    doc.add_paragraph(
        "V1 PT2 vẫn có điểm cao nhất và MRR@5 tăng lên 0.6603. Điều này cho thấy kết quả chính không chỉ đến từ các query được đánh dấu partly supported."
    )

    doc.add_heading("8. Nhận xét và giới hạn", level=1)
    bullet(doc, "Sau chỉnh sửa, đánh giá đã dùng query người dùng thực tế với quy trình nhãn có thể kiểm tra và tái lập, thay vì một bộ query nhỏ không mô tả rõ ground truth.")
    bullet(doc, "V1 PT2 đạt điểm quan sát cao nhất ở cả tập chính và tập supported-only. CLIP Image đơn lẻ có hiệu quả thấp với dạng truy vấn plot/quote; kết quả này phù hợp với quan sát định tính trước đó.")
    bullet(doc, "Không nên viết rằng V1 PT2 vượt BM25 hoặc SBERT có ý nghĩa thống kê: CI 95% của chênh lệch MRR@5 V1 PT2 − BM25 là [-0.0515, 0.1094] ở tập chính và [-0.0541, 0.1187] ở supported-only, đều chứa 0.")
    bullet(doc, "Trong paper nên dùng cụm 'user-reviewed target labels'. Để gọi là independently adjudicated ground truth, cần một reviewer thứ hai gán nhãn độc lập và báo cáo mức độ đồng thuận giữa các reviewer.")
    bullet(doc, "Mọi lựa chọn kiến trúc hoặc trọng số tiếp theo phải được chọn trên dev set; test set chỉ dùng cho báo cáo cuối cùng.")

    doc.add_heading("9. Tệp tái lập", level=1)
    for item in [
        "groundtruth/generated/qrels.tsv — nhãn relevance theo query và phim target",
        "groundtruth/generated/audit.csv — toàn bộ query, nhãn và quyết định review",
        "groundtruth/generated/split.csv — phân chia development/test theo người dùng",
        "groundtruth/generated/metrics_all.json — kết quả trên toàn bộ query hợp lệ",
        "groundtruth/generated/metrics_supported.json — kết quả chỉ trên query supported",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
