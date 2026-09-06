# Kết quả Đánh giá V3 (Image Captions)

## Bảng So Sánh MRR Từng Câu

### NHÓM 1: TRÍCH DẪN CHÍNH XÁC
| Query                             |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   Caption_SBERT |   PT3 |
|:----------------------------------|-------:|------------:|--------:|-------------:|----------------:|------:|
| I am going to make him an offe... |   1    |           1 |       0 |         0.25 |            0    |  1    |
| My mama always said life was l... |   0.5  |           1 |       1 |         0    |            0    |  0.5  |
| I ate his liver with some fava... |   1    |           0 |       1 |         0.33 |            0.33 |  1    |
| Keep your friends close, but y... |   0.25 |           0 |       0 |         0    |            0    |  0.25 |
| Here's looking at you, kid...     |   1    |           0 |       0 |         0    |            0    |  1    |

### NHÓM 2: NGỮ NGHĨA & CỐT TRUYỆN
| Query                             |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   Caption_SBERT |   PT3 |
|:----------------------------------|-------:|------------:|--------:|-------------:|----------------:|------:|
| two completely opposite famili... |    0   |         1   |       1 |         0    |            1    |     1 |
| a computer hacker learning the... |    1   |         0.5 |       1 |         0    |            0    |     1 |
| a banker wrongly convicted of ... |    0.5 |         0   |       1 |         0.2  |            0.33 |     1 |
| entering dreams to steal infor... |    0.5 |         0   |       1 |         0.25 |            1    |     1 |
| two gangsters, a boxer, and a ... |    0   |         1   |       1 |         0    |            0    |     1 |
| a girl trying to save her pare... |    1   |         0   |       1 |         0    |            0.2  |     1 |
| brother and sister struggling ... |    0   |         0   |       1 |         0.5  |            0.2  |     1 |

### NHÓM 3: HÌNH ẢNH & BỐI CẢNH
| Query                             |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   Caption_SBERT |   PT3 |
|:----------------------------------|-------:|------------:|--------:|-------------:|----------------:|------:|
| a woman screaming in a motel s... |   0.25 |           1 |     0.2 |            0 |            0.25 |  0.25 |
| a glowing mechanical suit flyi... |   0.5  |           0 |     1   |            0 |            0.5  |  0.5  |
| giant robots fighting monsters... |   0.5  |           0 |     0.5 |            0 |            1    |  1    |
| seven warriors defending a vil... |   1    |           0 |     1   |            0 |            0.5  |  1    |
| a dark knight standing on a ta... |   1    |           1 |     1   |            0 |            1    |  1    |

### NHÓM 4: BẪY TỪ VỰNG & SAI LỆCH
| Query                             |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   Caption_SBERT |   PT3 |
|:----------------------------------|-------:|------------:|--------:|-------------:|----------------:|------:|
| a guy with short term memory l... |    1   |           1 |       1 |          0   |               1 |     1 |
| two magicians competing and sa... |    0.2 |           0 |       1 |          0   |               0 |     1 |
| I am going to make him an offe... |    1   |           1 |       0 |          0.2 |               0 |     1 |

## Tổng kết Điểm số đa chiều

| Luồng Mô Hình   |    MRR |   Hit@1 |   Hit@5 |
|:----------------|-------:|--------:|--------:|
| BM25            | 0.61   |    0.45 |    0.85 |
| CLIP_Text       | 0.425  |    0.4  |    0.45 |
| SBERT           | 0.735  |    0.7  |    0.8  |
| CLIP_Image      | 0.0867 |    0    |    0.3  |
| Caption_SBERT   | 0.3658 |    0.25 |    0.6  |
| 🚀 PT3           | 0.875  |    0.8  |    1    |

## Confusion Matrix: Intent kỳ vọng vs Intent router dự đoán

(Không tính Nhóm 4 -- nhóm bẫy từ vựng cố tình không có 1 đáp án đúng)

| Kỳ vọng      |   exact quote |   movie plot |   visual scene |
|:-------------|--------------:|-------------:|---------------:|
| exact quote  |             5 |            0 |              0 |
| movie plot   |             0 |            6 |              1 |
| visual scene |             0 |            1 |              4 |

Độ chính xác router: 15/17 = 88.24%
