# Ground truth cho Movie Search — đề xuất dựa trên bài báo và khảo sát

Ngày kiểm tra: 15/09/2026. Trạng thái: **thiết kế và bộ biểu mẫu, chưa có nhãn đã xác minh**.

## 1. Kết luận và phạm vi

Nên dùng **tìm lại phim người dùng đang nhớ (known-item movie retrieval)** làm tác vụ chính: đầu vào là câu nhớ lại của người dùng; đầu ra là danh sách phim không trùng; đáp án mục tiêu là phim người dùng xác nhận muốn tìm. Đây là tác vụ sát với form khảo sát và code hiện tại. Nghiên cứu về tip-of-the-tongue movie identification là related work trực tiếp phù hợp [1]. Tuy nhiên, form yêu cầu người dùng chọn phim trước rồi viết ba câu là **khảo sát có hướng dẫn**, không phải log tìm kiếm tự nhiên hay bằng chứng họ thực sự quên tên phim.

Thêm lớp xác minh nội dung/cảnh để biết query được hỗ trợ bởi phim gốc và bởi dữ liệu đã index đến đâu. Nếu muốn tuyên bố hệ thống tìm được *mọi phim phù hợp với mô tả*, phải xây thêm nhãn relevance nhiều phim. Nếu muốn tuyên bố tìm đúng *cảnh*, phải đánh giá riêng với scene ID/timestamp. Không suy ra hai kết quả đó từ việc trả đúng tên phim.

Reviewer không quy định ngưỡng “180 câu”, tỷ lệ chia tập, số annotator hoặc thang nhãn bắt buộc. Các con số bên dưới là thiết kế đề xuất cho dự án; không bảo đảm bài được chấp nhận.

## 2. Những gì đã kiểm tra

| Hạng mục | Kết quả |
|---|---|
| File khảo sát | 61 dòng trả lời không rỗng; 183 ô query có nội dung; 61 câu mỗi nhóm |
| Người tham gia duy nhất | Chưa xác minh; nickname và dòng trả lời không đủ chứng minh 61 người khác nhau |
| Phim người trả lời chọn | 29 tên phim khác nhau |
| Metadata GitHub | 62 tên phim khác nhau trong `movies_data_english_clean.json`; cả 29 tên khảo sát đều khớp chính xác |
| Corpus bài báo | Báo cáo 60 phim, 38.202 frame và 38.202 subtitle chunk; chưa xác minh bằng index/video thực tế |
| Evaluation trong bài/code V1 | 20 query: 5 quote, 7 plot, 5 visual, 3 trap; `data/queries.json` có 40 câu nhưng evaluator V1 hard-code 20 câu |
| Đầu ra được chấm | Tên phim; code gộp các kết quả về phim không trùng |
| Trùng sau chuẩn hóa phục vụ audit | 3 cụm quote trùng; 179 nội dung khác nhau tính trong từng nhóm, trên 183 lượt query |

Chuẩn hóa audit: Unicode NFKC, casefold, chỉ giữ chữ/số. **Không dùng bản chuẩn hóa thay đầu vào retrieval**. Các dòng Excel trùng: 5/61, 8/35/60, 10/24. Chưa rà hết câu diễn đạt gần nghĩa.

Ví dụ cần kiểm tra thủ công:

- Dòng Excel 54 chứa tên phim ngay trong cả ba câu và nội dung có dấu hiệu không trả lời yêu cầu. Đề xuất xem xét loại theo tiêu chí chất lượng định trước, không loại chỉ vì cần đúng 60 người.
- Dòng Excel 42, nhóm quote: `I can't remember.` là ứng viên non-response, cần xác nhận. Nhóm plot `Quiet yet powerful.` rất mơ hồ; không tự động kết luận vô nghĩa.
- Câu visual `a giant robot is fighting with a monster` có thể phù hợp nhiều phim. Phim người dùng chọn vẫn là target của known-item, nhưng không chứng minh những phim khác đều không liên quan.
- Quote có biến thể, lỗi nhớ hoặc lỗi chính tả. Vì vậy tên nhóm nên là **Remembered dialogue/quotes**, không gọi toàn bộ là **Exact quotes** khi chưa xác minh.

## 3. Chốt 60 người / 180 query đúng cách

1. Giữ nguyên XLSX gốc; lưu hash và đánh mã giả P001… theo dòng phản hồi. Không đưa nickname/timestamp vào bản phát hành đánh giá.
2. Xác nhận mỗi mã là một người; ghi cách tuyển, có phải tác giả hay không, hướng dẫn, ngôn ngữ, có dịch/tra cứu/AI hỗ trợ hay không nếu đã thu thập được. Không suy luận cách tạo câu từ văn phong. Chỉ tuyên bố điều có bằng chứng.
3. Quy định trước khi xem điểm mô hình: loại phản hồi rỗng, spam, không thực hiện tác vụ, trùng người; ghi lý do từng quyết định. Không loại vì query khó, nhớ sai, ngắn hoặc mô hình không tìm thấy.
4. Phân biệt loại cả phản hồi với loại một query. Nếu loại dòng 54 sau kiểm tra thì còn **60 phản hồi / 180 query thô**, chưa chắc có 180 query dùng được. Ví dụ nếu `I can't remember.` được xác nhận là không cung cấp query, chỉ còn 179 câu dùng được, trước các kiểm tra khác.
5. Nếu cần đủ 180 câu dùng được, nhờ chính người tham gia bổ sung phần thiếu bằng trí nhớ của họ, ghi phiên bản/thời điểm, trước khi khóa test. Nếu không thể, báo cáo số thực tế hoặc tuyển bổ sung theo quy tắc đã định. Không tự viết thay và ghi là câu khảo sát.
6. Câu trùng từ những người độc lập có thể giữ trong đánh giá theo phân bố phản hồi. Báo cáo thêm kết quả bỏ trùng để xem độ nhạy. Không gọi 180 lượt query là 180 câu độc lập hoặc 180 nội dung khác nhau.

## 4. Khóa corpus trước khi gán nhãn cuối

### Cập nhật kiểm tra local và xử lý query

Máy có Docker CLI và `/Applications/Docker.app`, nhưng lệnh `docker info` ngoài sandbox không kết nối được daemon tại socket đang cấu hình. `docker-compose.yml` chỉ định nghĩa ChromaDB; không có service cắt frame hay chạy Whisper. Trong bản clone hiện không có `data/DataMovie/` và `ChromaDB_Storage/`. `.gitignore` loại `data/DataMovie/` khỏi Git. Không tìm thấy `ffmpeg`/`ffprobe` trên PATH hiện tại; điều này không loại trừ bản cài ở môi trường khác.

`db_builder.py` đọc dữ liệu đã trích xuất từ `data/DataMovie/<movie>/picture/` và `script/*.txt`, rồi tạo embedding/index. Không tìm thấy pipeline tải video, chạy FFmpeg và Whisper trong các file code/notebook của repo đã kiểm tra. `PictureToText.ipynb` tạo caption BLIP-2 từ ảnh có sẵn trên Google Drive, không cắt frame từ movie. Đường dẫn trong notebook gợi ý dữ liệu từng nằm ở `MyDrive/Studies/Data_Retrieval/Data_extractor/DataMovie`; chưa xác nhận thư mục này hiện có hoặc truy cập được.

Thứ tự khôi phục: lấy lại dữ liệu trích xuất và code extractor gốc → kiểm kê/mapping phim/frame/transcript/timestamp → khởi động và xác nhận ChromaDB → xây index. Chỉ khởi động Docker không khôi phục media hoặc vector cũ.

Không chạy chức năng rebuild ngay khi thiếu dữ liệu: `clean_and_translate()` có thể ghi đè JSON sạch thành danh sách rỗng vì không có folder khớp; `build_vector_db()` xóa các collection hiện hữu trước khi xây lại. Ngoài ra, code V1 tạo BM25 từ `bm25_docs` chỉ gồm dialogue nhưng lưu cùng `txt_metas`/`docs` có cả summary, gây nguy cơ ánh xạ sai index–phim; cần sửa/kiểm chứng trước chạy benchmark mới. Chưa kết luận file pickle hiện có được tạo bởi đúng phiên bản code này.

Về nội dung query, bản cập nhật giữ `query_raw` và thêm `query_clean`, `cleanup_actions`, `content_review_status`. Làm sạch kỹ thuật gồm Unicode NFC, bỏ một số ký tự ẩn, gộp khoảng trắng/xuống dòng và trim; 5/183 câu thay đổi. Không thay chữ, sửa chính tả, dịch, bổ sung tên nhân vật hoặc viết lại theo phim mục tiêu. Toàn bộ nhãn kiểm chứng nội dung vẫn pending.

Quy trình nội dung tiếp theo: kiểm tra câu có thực hiện tác vụ không → kiểm tra intent thực tế → xác minh target/phiên bản → đối chiếu quote/plot/visual với nguồn → phân loại nhớ đúng, đúng một phần, mâu thuẫn hoặc chưa đủ bằng chứng. Nếu muốn thử hệ thống sửa lỗi/ngữ pháp hoặc diễn đạt lại query, lưu thành `query_rewritten` riêng với lý do và người duyệt; không dùng target hay nhãn đúng làm đầu vào bộ sửa query. Đánh giá raw và rewritten thành hai điều kiện riêng, áp dụng cùng cách xử lý cho các mô hình. Khóa lựa chọn raw/clean đầu vào chính trước khi chạy test; mặc định dùng raw để giữ nguyên khảo sát.

Tạo manifest gồm `movie_id`, title, năm/phiên bản, định danh nguồn, hash hoặc version, số frame và chunk thực tế, các mốc thời gian. ID trong bộ biểu mẫu được tạo từ title/năm metadata, chỉ là ID ứng viên, cần xác nhận đúng bản phim.

- Kiểm tra trực tiếp ChromaDB, BM25 và media để thống nhất corpus thực chạy: 60 hay 62 phim; có phim chỉ có metadata mà thiếu video/frame không?
- Chốt một corpus chung cho mọi hệ thống. Các phim không được người dùng chọn vẫn nằm trong tập tìm kiếm để làm distractor; không giới hạn tìm kiếm còn 29 target.
- Nếu target ngoài corpus: đánh dấu `out_of_corpus`, báo số riêng. Benchmark closed-corpus chỉ chấm tập hợp lệ theo quy tắc đã công bố. Không biến target ngoài corpus thành phim khác; không gọi kết quả ngoài corpus là lỗi ranking thông thường.
- Báo riêng số phim, frame, subtitle chunk, và **số cặp thực sự được nối theo timestamp**. Hai bảng cùng có 38.202 phần tử không tự chứng minh quan hệ đồng bộ 1–1.
- Synopsis là tài liệu được index, không mặc nhiên là nguồn đáp án đáng tin. Cần kiểm chứng nội dung, đặc biệt phiên bản/remake/phần tiếp theo.

TREC nhấn mạnh corpus và qrels phải tương ứng [2].

## 5. Ba lớp nhãn không được trộn

### A. Target phim — bắt buộc cho tác vụ chính

Mỗi query lưu `participant_id`, nguyên văn query, `prompt_group`, `target_movie_id`, trạng thái xác nhận target và eligibility. Cột phim người dùng chọn là **self-reported target**, chỉ chuyển thành target cuối khi danh tính/phiên bản đã được xác minh; khi mơ hồ cần người dùng xác nhận.

Với tác vụ này, một phim tương tự nhưng không phải target không phải kết quả thành công. Đây là định nghĩa thành công của known-item, **không phải kết luận phim đó không liên quan về nội dung**. Ký ức sai một chi tiết không tự làm mất target đã xác nhận: giữ và gắn cờ để phân tích độ bền vững với trí nhớ không hoàn hảo.

### B. Nội dung và bằng chứng — bắt buộc cho phân tích lỗi multimodal

Hai người đánh giá độc lập ghi:

- Intent thực tế: quote / plot / visual / mixed / unclear. Giữ riêng `prompt_group` do form đặt ra. Một query nằm cột visual chưa chắc thật sự chỉ có thông tin thị giác. Không dùng tên cột làm nhãn đúng của router.
- Query được phim hỗ trợ: supported / partly_supported / contradicted / unclear.
- Quote: verbatim / approximate / not_found / not_applicable / unknown.
- Cảnh trong phim gốc: present / absent / not_applicable / unknown.
- Bằng chứng trong frame và text đã index, mỗi loại: present / absent / not_applicable / unknown.
- Đường dẫn nguồn, timestamp/đoạn, frame ID, transcript ID, ghi chú đối chiếu. Với plot toàn phim, có thể dùng synopsis đã kiểm chứng và nhiều đoạn phim; không ép gắn vào một cảnh.

`unknown` nghĩa chưa xác minh, không phải nhãn âm. Whisper có thể sai; dùng audio/video gốc kiểm tra câu thoại, không dùng output Whisper làm trọng tài duy nhất.

Ví dụ cách diễn giải, **chưa phải nhãn đã gán cho dữ liệu này**: cảnh có trong phim nhưng không có trong frame đã lấy mẫu → vấn đề coverage/extraction; có frame hỗ trợ nhưng xếp hạng sai → vấn đề retrieval/fusion. Giữ cả hai trong điểm end-to-end; thêm bảng trên tập có evidence để chẩn đoán, ghi rõ mẫu số.

### C. Relevance nhiều phim và relevance cảnh — bổ sung khi có claim tương ứng

Nếu đánh giá mọi phim phù hợp với mô tả, gán nhãn query–movie riêng:

| Grade | Quy tắc |
|---|---|
| 3 | Đáp ứng rõ các chi tiết chủ chốt và ràng buộc được nêu, có bằng chứng |
| 2 | Phù hợp đáng kể với nhu cầu chính, thiếu một số chi tiết phụ |
| 1 | Chỉ có yếu tố chung hoặc liên hệ yếu; chưa đáp ứng nhu cầu chính |
| 0 | Không phù hợp, sau khi đã xem bằng chứng đủ để kết luận |
| Để trống | Chưa gán nhãn; tuyệt đối không tự điền 0 |

Grade không được phụ thuộc vào việc có đúng phim người dùng chọn hay không. Query rộng có thể có nhiều grade 3. Ngưỡng relevance nhị phân đề xuất là grade ≥ 2 và khóa trước đánh giá. Đây là rubric của dự án, không phải thang do reviewer/TREC bắt buộc.

Với corpus 60 phim, cách ít lệ thuộc hệ thống nhất là xét toàn bộ query–movie: 180 × 60 = **10.800 cặp**, hai người = **21.600 lượt chấm**, chưa tính phân xử. Nếu corpus 62 phim thì là 11.160 cặp. Đây là khối lượng cho **lớp nhiều phim tùy chọn**, không phải yêu cầu tối thiểu của known-item.

Nếu không đủ nhân lực: tạo pool là hợp của top-10 phim không trùng từ BM25, SBERT, CLIP image, CLIP text, RRF và các hệ thống fusion/ablation định đánh giá; thêm target và một số phim ngẫu nhiên ngoài pool. Lưu nguồn pool riêng, xáo thứ tự và ẩn tên hệ thống/rank/target khi chấm general relevance. Có thể mở rộng top-20 theo quy tắc đã định; báo tỷ lệ có nhãn tại cutoff và số positive mới khi mở rộng. Pooling là kỹ thuật lập tập đánh giá IR đã được TREC sử dụng [2,3], nhưng không bảo đảm tìm đủ mọi phim liên quan.

Không diễn giải Recall tính trên positive trong pool thành recall đầy đủ toàn corpus. Khi so sánh hệ thống mới, chấm thêm các kết quả chưa được xét; không gán nhãn âm hàng loạt cho tài liệu chưa chấm. nDCG cũng chịu ảnh hưởng nhãn thiếu; cần công bố độ phủ và cách xử lý.

Nếu đánh giá cảnh: thêm scene ID, `start_seconds`, `end_seconds`, source edition và liên kết frame/chunk. Quy định trước thế nào là cùng cảnh hoặc mức overlap chấp nhận. Phải sửa evaluator để nhận các scene ID/timestamp: code trả tên phim hiện không đánh giá được temporal localization. Không cần gán tất cả 38.202 frame cho mọi query để đánh giá movie retrieval.

## 6. Quy trình annotator

1. Dùng 2 người chấm độc lập; 1 người thứ ba phân xử bất đồng. Nêu vai trò, kinh nghiệm và quan hệ với nhóm tác giả. Người tạo query không là người duy nhất xác nhận relevance.
2. Pilot khoảng 12–18 query, trải đều ba nhóm, trên tập dev; thảo luận điểm khó và sửa hướng dẫn trước khi khóa. Nếu đổi rubric, chấm lại các mục chịu ảnh hưởng, không ghi đè lịch sử nhãn.
3. Hai người chấm intent từ query trước khi thấy cột nhóm và target. Sau đó kiểm tra target/evidence. Với general relevance, ẩn cả target, rank và hệ thống.
4. Giữ nhãn A/B ban đầu; lưu nhãn phân xử và lý do riêng. Không cho A sao chép B; không dùng điểm mô hình giải quyết bất đồng.
5. Trước phân xử: báo số mục cả hai đã chấm, tỷ lệ đồng thuận; Cohen’s κ cho intent/nhãn danh mục, weighted κ nếu dùng grade thứ bậc, kèm phân bố nhãn và tỷ lệ thiếu. Không dùng tỷ lệ 100% sau phân xử làm agreement; nếu κ không xác định do chỉ có một lớp thì ghi N/A. Không đặt một ngưỡng κ tùy ý làm bằng chứng bảo đảm chất lượng.
6. AI có thể hỗ trợ chuẩn hóa, phát hiện trùng và tổ chức evidence; nhãn test cuối cần được người đánh giá kiểm chứng. Không tuyên bố nhãn AI là nhãn người [5].

Known-item với 180 query cần tối thiểu 360 lượt kiểm tra target/evidence nếu hai người chấm toàn bộ, chưa tính pilot và phân xử. Ước lượng thời gian bằng pilot; không giả định mọi cảnh đều xác minh trong vài phút.

## 7. Held-out test và thống kê

### Thiết kế đề xuất nếu tổng cộng chỉ có 60 người

Sau QC, chia theo người: khoảng **15 người dev (45 query)** và **45 người test (135 query)**, mỗi nhóm có 15 dev / 45 test nếu mọi người đủ ba câu. Đây là số mục tiêu, chưa phải split đã tạo.

- Ba câu của cùng một người luôn cùng tập. Cụm câu trùng hoặc gần trùng giữa người dùng cũng cần cùng tập khi có nguy cơ rò rỉ; lập nhóm liên kết trước khi chia, ghi lại seed và cân bằng nhóm/phim trong khả năng cho phép. Cụm lớn có thể làm tỷ lệ thực tế lệch mục tiêu.
- Kiểm tra trùng với query cũ đã dùng phát triển; tránh coi biến thể gần giống là test sạch. Nếu không thể tránh, báo overlap và kết quả sensitivity trên subset không overlap.
- Tinh chỉnh trọng số, boost, prompt router, cutoff và chọn mô hình chỉ trên dev. Giữ nhãn test khỏi người phát triển; khóa model/config rồi đánh giá test. Việc đọc dữ liệu để QC không đồng nghĩa đã tune trên test, nhưng cần ghi trung thực quá trình truy cập và phát triển trước đó.
- **180 là tổng dev + test**, không phải số test. Nếu muốn cả 180 là test, cần tập dev riêng không rò rỉ với 60 người này và khóa hệ thống trước thử nghiệm; bộ query cũ chỉ phù hợp nếu kiểm tra overlap và lịch sử sử dụng.
- Đây là test trên người/query mới trong cùng corpus; không tuyên bố tổng quát sang phim chưa thấy. Nếu muốn claim đó, cần split/thiết kế riêng theo phim.

### Metrics

Chính: **MRR@5**, **Success@1**, **Success@5** với một target, rank tính trên phim không trùng. MRR@5 = trung bình `1/rank` nếu target trong top 5, ngược lại 0. Để so sánh bài cũ, đặt đúng tên cutoff: code chỉ lấy top 5 nên số gọi “MRR” thực chất là MRR@5.

Với một target, Recall@5 = Success@5 và Precision@5 = Success@5 / 5; Precision@5 tối đa 0,2. Do đó không cần dùng ba con số phụ thuộc nhau như ba bằng chứng độc lập. Nếu báo MRR không cutoff, phải lấy ranking đầy đủ.

Tùy chọn nhiều phim: nDCG@5 với gain `2^grade - 1`; P@5/MRR@5 với relevance ≥ 2, ghi rõ khác target task. Chỉ gọi recall toàn corpus khi nhãn đủ hỗ trợ. Khóa quy tắc cho query không có positive, không im lặng bỏ chúng khỏi mẫu số.

Bảng kết quả gồm tổng test và từng nhóm: số người, số query, số target khác nhau, số lần hit/total, MRR@5 và CI 95%. Thêm bảng per-movie hoặc macro theo phim có query, không gán điểm 0 cho phim không có query. Báo riêng cả ba prompt group và nhãn intent thực tế khi cần.

CI đề xuất: paired cluster bootstrap 10.000 lần, lấy mẫu **người dùng** với hoàn lại và giữ nguyên các query của người đó; dùng cùng mẫu cho mọi hệ thống. Báo CI cho điểm và chênh lệch so với baseline. Không bootstrap 180 query như thể độc lập. Nếu câu trùng tạo phụ thuộc mạnh giữa nhiều người, thêm phân tích theo cụm trùng hoặc kết quả bỏ trùng; diễn giải CI có điều kiện trên corpus này.

So sánh chính khóa trước: fusion so với SBERT trên MRR@5. Có thể dùng paired permutation test với đổi dấu chênh lệch ở cấp người; nếu kiểm định nhiều baseline/cutoff, chỉnh multiple comparisons (ví dụ Holm) và báo cả effect size. Nghiên cứu IR hỗ trợ dùng kiểm định paired và khảo sát bootstrap/randomization [4]; lựa chọn cluster theo người là điều chỉnh cho thiết kế ba query/người ở đây.

Không coi 180 là đảm bảo statistical power. Dùng pilot/dev để ước lượng độ biến động và hiệu ứng có thể phát hiện; nếu CI rộng hoặc chưa khác biệt rõ thì báo đúng, không tiếp tục thay nhãn/chọn query để fusion thắng.

## 8. Đối chiếu trực tiếp với review và sửa bài

| Phê bình | Bằng chứng cần bổ sung |
|---|---|
| ~20 query quá ít | Sơ đồ số phản hồi thô → loại/bổ sung → tập cuối → dev/test; số tuyệt đối, CI và giới hạn mẫu |
| Ground truth không rõ | Tác vụ/đơn vị chấm; target do người dùng cung cấp; hướng dẫn, bằng chứng, A/B độc lập và phân xử |
| Tác giả biết corpus | Mô tả tuyển người và cách elicitation; không cho xem kết quả hệ thống; disclose corpus-constrained survey và overlap với query cũ |
| Dataset/evaluation mismatch | Manifest thực tế; số query và 29 target hiện có; tìm kiếm toàn corpus; coverage theo phim, không tuyên bố đủ 60 phim đã được hỏi |
| Thiếu held-out/thống kê | Split theo người/cụm trùng; tune trên dev; test khóa; MRR@5/Success@k, paired CI |
| Related work lệch | Thêm known-item movie retrieval [1], test collections/relevance [2,3], significance [4]; giữ nguồn gốc BM25/SBERT/CLIP/RRF phù hợp |

Trong PDF hiện tại, sửa Section 3.1 và 4.1.2: frame/transcript tự động là **retrieval corpus / indexed representations**, không phải “ground truth ... without human intervention”. Section 5.4 không nên khẳng định Whisper đảm bảo “perfect temporal alignment” nếu chưa đo sai số thời gian.

Không khẳng định multimodal “cần thiết”, “ổn định hơn” hoặc “ưu việt” chỉ từ việc CLIP image yếu; bảng hiện tại có SBERT tốt hơn fusion. Ground truth tốt nhằm đánh giá trung thực, không bảo đảm fusion thắng. Với hệ thống mới, dùng cùng test cho các ablation: text-only, visual-only, RRF, thêm reranker, thêm static boost, adaptive fusion nếu có. Router nhận query text, không nhận nhãn nhóm khảo sát; oracle routing chỉ báo như upper-bound riêng.

PDF được cung cấp có tiêu đề khác tiêu đề trong thư review. Bản nhận xét nhắc một số lĩnh vực references không trùng hoàn toàn danh mục PDF hiện tại; cần ghi rõ version bản sửa và bản đã nộp, không giả định mọi nhận xét đều trỏ cùng bản.

## 9. Sử dụng bộ file đã chuẩn bị

- `generated/groundtruth_coordinator_DRAFT.xlsx`: query nguyên văn, mã phản hồi, cờ QC, metadata phim ứng viên, coverage và sheet phân xử. Các nhãn cuối để trống.
- `generated/annotator_A_DRAFT.xlsx`, `annotator_B_DRAFT.xlsx`: hai biểu mẫu độc lập, thứ tự query được xáo; sheet Intent không có prompt group/target.
- `generated/queries_DRAFT.csv`: 183 query với target do người dùng tự khai và trạng thái `unverified`, không phải qrels cuối.
- `generated/audit.json`: thống kê, hash nguồn, cụm trùng; số verified labels hiện bằng 0.
- `build_annotation_pack.py`: tạo lại các biểu mẫu từ nguồn. **Copy file đã điền ra ngoài `generated/` trước khi chạy lại**, vì script ghi đè bản draft.

Chưa tạo qrels cuối, chưa tự quyết loại người, chưa chia dev/test. Khi QC và hai lượt kiểm chứng hoàn tất, xuất `queries.tsv`, `qrels_target.tsv` (dạng `query_id 0 movie_id 1` cho target đã chốt), nhãn relevance riêng nếu có, manifest corpus và split/version. Không xuất nhãn `unknown` thành 0. Lưu prediction/score riêng khỏi labels.

## 10. Nguồn tham khảo đã đối chiếu

1. Arguello et al. (2021), [Tip of the Tongue Known-Item Retrieval: A Case Study in Movie Identification](https://arxiv.org/abs/2101.07124). Liên quan trực tiếp đến tìm lại phim từ trí nhớ.
2. NIST, [TREC English Relevance Judgements](https://trec.nist.gov/data/reljudge_eng.html). Relevance judgments, pooling và yêu cầu corpus–qrels tương ứng; không áp nguyên định nghĩa relevance của tìm tài liệu vào known-item.
3. NIST, [TREC-COVID Round 4 Task Guidelines](https://ir.nist.gov/trec-covid/round4.html). Ví dụ phương pháp pooling khi không thể chấm toàn corpus; dẫn cho methodology đánh giá, không làm related work về mô hình phim.
4. Smucker, Allan & Carterette (2007), [A Comparison of Statistical Significance Tests for Information Retrieval Evaluation](https://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=744). Nghiên cứu kiểm định paired cho so sánh hệ thống IR.
5. Soboroff (2025), [Don't Use LLMs to Make Relevance Judgments](https://www.nist.gov/publications/dont-use-llms-make-relevance-judgments). Thảo luận rủi ro của dùng nhãn LLM thay đánh giá relevance của con người.
