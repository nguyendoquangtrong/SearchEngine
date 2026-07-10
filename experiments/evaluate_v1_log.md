# Kết quả Đánh giá V1

## Bảng So Sánh MRR Từng Câu

### NHÓM 1: TRÍCH DẪN CHÍNH XÁC
| Query                             |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   PT2 |
|:----------------------------------|-------:|------------:|--------:|-------------:|------:|
| I am going to make him an offe... |   1    |           0 |       0 |         0.25 |   0   |
| My mama always said life was l... |   0.5  |           0 |       1 |         0    |   0   |
| I ate his liver with some fava... |   1    |           0 |       0 |         0.33 |   0.5 |
| Keep your friends close, but y... |   0.25 |           0 |       0 |         0    |   0   |
| Here's looking at you, kid...     |   1    |           0 |       0 |         0    |   0   |

### NHÓM 2: NGỮ NGHĨA & CỐT TRUYỆN
| Query                             |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   PT2 |
|:----------------------------------|-------:|------------:|--------:|-------------:|------:|
| two completely opposite famili... |    0   |         1   |       1 |         0    |     1 |
| a computer hacker learning the... |    1   |         0.5 |       1 |         0    |     0 |
| a banker wrongly convicted of ... |    0.5 |         1   |       1 |         0.2  |     1 |
| entering dreams to steal infor... |    0.5 |         0   |       1 |         0.25 |     1 |
| two gangsters, a boxer, and a ... |    0   |         1   |       1 |         0    |     1 |
| a girl trying to save her pare... |    1   |         0   |       1 |         0    |     0 |
| brother and sister struggling ... |    0   |         0.5 |       1 |         0.5  |     1 |

### NHÓM 3: HÌNH ẢNH & BỐI CẢNH
| Query                             |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   PT2 |
|:----------------------------------|-------:|------------:|--------:|-------------:|------:|
| a woman screaming in a motel s... |   0.25 |           0 |     0.2 |            0 |   0   |
| a glowing mechanical suit flyi... |   0.5  |           0 |     1   |            0 |   0.2 |
| giant robots fighting monsters... |   0.5  |           0 |     0.5 |            0 |   0   |
| seven warriors defending a vil... |   1    |           0 |     1   |            0 |   1   |
| a dark knight standing on a ta... |   1    |           1 |     1   |            0 |   1   |

### NHÓM 4: BẪY TỪ VỰNG & SAI LỆCH
| Query                             |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   PT2 |
|:----------------------------------|-------:|------------:|--------:|-------------:|------:|
| a guy with short term memory l... |    1   |           1 |       1 |          0   |     1 |
| two magicians competing and sa... |    0.2 |           0 |       1 |          0   |     1 |
| I am going to make him an offe... |    1   |           0 |       0 |          0.2 |     0 |

## Tổng kết Điểm số đa chiều

| Luồng Mô Hình   |    MRR |   Precision@5 |   Recall@5 |
|:----------------|-------:|--------------:|-----------:|
| BM25            | 0.61   |          0.17 |       0.85 |
| CLIP_Text       | 0.3    |          0.07 |       0.35 |
| SBERT           | 0.685  |          0.15 |       0.75 |
| CLIP_Image      | 0.0867 |          0.06 |       0.3  |
| 🚀 PT2           | 0.485  |          0.11 |       0.55 |
