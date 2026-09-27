# Kết quả Đánh giá V1 trên Ground Truth

- **Thời gian chạy**: 2026-09-25 16:47:26
- **Tập dữ liệu**: `groundtruth/generated/split.csv`
- **Cấu hình**: Split = `TEST` | Lọc hỗ trợ = `all` | Top-N = `5`
- **Quy mô**: 132 câu truy vấn hợp lệ từ 45 người tham gia duy nhất.

## 1. Tổng kết Điểm số Đa chiều Toàn cục (Overall Benchmark)

| Hệ thống                           |   Success@1 |   Success@5 |   MRR@5 |   Precision@5 | 95% CI (MRR, Clustered)   |
|:-----------------------------------|------------:|------------:|--------:|--------------:|:--------------------------|
| BM25                               |      0.4773 |      0.7424 |  0.5652 |        0.1485 | [0.4917, 0.6369]          |
| CLIP_Text                          |      0.2955 |      0.3485 |  0.3197 |        0.0697 | [0.2308, 0.4122]          |
| SBERT                              |      0.4848 |      0.6061 |  0.5362 |        0.1212 | [0.4545, 0.6174]          |
| CLIP_Image                         |      0.0758 |      0.3561 |  0.1687 |        0.0712 | [0.1088, 0.2332]          |
| 🚀 SearchEngine_V1 (Multimodal PT2) |      0.5833 |      0.7955 |  0.6606 |        0.1591 | [0.5741, 0.7433]          |

## 2. Điểm số Phân rã theo Từng Nhóm Khảo sát (Prompt Groups)

| Nhóm                                       |   Số câu | Hệ thống                |   Success@1 |   Success@5 |   MRR@5 |
|:-------------------------------------------|---------:|:------------------------|------------:|------------:|--------:|
| Nhóm 1: Trích dẫn thoại (Quotes)           |       43 | BM25                    |      0.7442 |      0.9302 |  0.8058 |
| Nhóm 1: Trích dẫn thoại (Quotes)           |       43 | CLIP_Text               |      0.2326 |      0.2326 |  0.2326 |
| Nhóm 1: Trích dẫn thoại (Quotes)           |       43 | SBERT                   |      0.3721 |      0.4884 |  0.4174 |
| Nhóm 1: Trích dẫn thoại (Quotes)           |       43 | CLIP_Image              |      0      |      0.3488 |  0.1174 |
| Nhóm 1: Trích dẫn thoại (Quotes)           |       43 | 🚀 SearchEngine_V1 (PT2) |      0.7209 |      0.7442 |  0.7267 |
| Nhóm 2: Ngữ nghĩa & Cốt truyện (Plot)      |       45 | BM25                    |      0.3556 |      0.6667 |  0.4559 |
| Nhóm 2: Ngữ nghĩa & Cốt truyện (Plot)      |       45 | CLIP_Text               |      0.2667 |      0.3111 |  0.2889 |
| Nhóm 2: Ngữ nghĩa & Cốt truyện (Plot)      |       45 | SBERT                   |      0.6444 |      0.7556 |  0.6944 |
| Nhóm 2: Ngữ nghĩa & Cốt truyện (Plot)      |       45 | CLIP_Image              |      0.1556 |      0.4222 |  0.2367 |
| Nhóm 2: Ngữ nghĩa & Cốt truyện (Plot)      |       45 | 🚀 SearchEngine_V1 (PT2) |      0.6444 |      0.8222 |  0.7167 |
| Nhóm 3: Hình ảnh & Bối cảnh (Visual Scene) |       44 | BM25                    |      0.3409 |      0.6364 |  0.4417 |
| Nhóm 3: Hình ảnh & Bối cảnh (Visual Scene) |       44 | CLIP_Text               |      0.3864 |      0.5    |  0.4364 |
| Nhóm 3: Hình ảnh & Bối cảnh (Visual Scene) |       44 | SBERT                   |      0.4318 |      0.5682 |  0.4905 |
| Nhóm 3: Hình ảnh & Bối cảnh (Visual Scene) |       44 | CLIP_Image              |      0.0682 |      0.2955 |  0.1492 |
| Nhóm 3: Hình ảnh & Bối cảnh (Visual Scene) |       44 | 🚀 SearchEngine_V1 (PT2) |      0.3864 |      0.8182 |  0.5386 |

## 3. Bảng Chi tiết MRR@5 Từng Câu Truy vấn

### NHÓM 1: TRÍCH DẪN THOẠI (QUOTES)
| ID      | Query                               | Target                                            |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   V1_PT2 |
|:--------|:------------------------------------|:--------------------------------------------------|-------:|------------:|--------:|-------------:|---------:|
| P001_G1 | Don’t leave me murph                | Interstellar                                      |   0.5  |           0 |    1    |         0.5  |     1    |
| P002_G1 | i’m afraid of the dark              | The Green Mile                                    |   0    |           0 |    0    |         0    |     0    |
| P004_G1 | If you run, the beast catches yo... | City Of God                                       |   0.2  |           0 |    0    |         0    |     0    |
| P005_G1 | I just wanted you to fight for m... | Real Steel                                        |   1    |           0 |    0    |         0    |     1    |
| P007_G1 | you mustn't be afraid to dream a... | Inception                                         |   1    |           0 |    0.2  |         0.25 |     1    |
| P009_G1 | I'm gonna make him an offer he c... | The Godfather                                     |   1    |           1 |    0    |         0.5  |     1    |
| P011_G1 | You either die a hero, or you li... | The Dark Knight                                   |   1    |           0 |    0.5  |         0    |     1    |
| P012_G1 | Hakuna Matata                       | The Lion King                                     |   1    |           0 |    1    |         0.5  |     1    |
| P013_G1 | Unfortunately, no one can be tol... | The Matrix                                        |   1    |           1 |    1    |         0    |     1    |
| P014_G1 | Here’s looking at you, kid.         | Casablanca                                        |   1    |           0 |    0    |         0    |     1    |
| P016_G1 | I am the Half-Blood Prince.         | Harry Potter and the Half-Blood Prince            |   0.33 |           1 |    1    |         0    |     1    |
| P017_G1 | I hope to see my friend and shak... | The Shawshank Redemption                          |   1    |           0 |    0    |         0    |     1    |
| P018_G1 | Frankly, my dear, I don't give a... | Gone with the Wind                                |   1    |           1 |    1    |         0.2  |     1    |
| P019_G1 | Once again, I must ask too much ... | Harry Potter and the Half-Blood Prince            |   1    |           0 |    1    |         0.5  |     1    |
| P020_G1 | I don't have to defend my decisi... | 12 Angry Men                                      |   0.33 |           0 |    0.5  |         0    |     1    |
| P023_G1 | I'm gonna make him an offer he c... | The Godfather                                     |   1    |           1 |    0.5  |         0.5  |     1    |
| P024_G1 | she is nice because she is rich     | Parasite                                          |   1    |           1 |    1    |         0    |     1    |
| P025_G1 | Come with me if you want to live.   | Terminator 2: Judgment Day                        |   1    |           0 |    1    |         0.5  |     1    |
| P026_G1 | earn this                           | Saving Private Ryan                               |   1    |           0 |    0    |         0    |     0    |
| P028_G1 | There is no spoon.                  | The Matrix                                        |   1    |           0 |    0    |         0    |     1    |
| P029_G1 | Remember who you are.               | The Lion King                                     |   1    |           0 |    0    |         0.2  |     1    |
| P030_G1 | We create our own demons.           | Iron Man 3                                        |   1    |           0 |    0    |         0    |     0    |
| P031_G1 | ​Life is like a box of chocolate... | Forrest Gump                                      |   1    |           1 |    1    |         0    |     1    |
| P034_G1 | You mustn't be afraid to dream a... | Inception                                         |   1    |           0 |    0.25 |         0.25 |     1    |
| P035_G1 | I am not a toy. I'm trash!          | Toy Story 4                                       |   1    |           0 |    0    |         0.25 |     1    |
| P038_G1 | I'm not a smart man, but I know ... | Forrest Gump                                      |   1    |           0 |    0    |         0    |     0    |
| P039_G1 | There has been an awakening. Hav... | Star Wars: The Force Awakens                      |   1    |           0 |    1    |         0    |     1    |
| P043_G1 | I am half-blood prince.             | Harry Potter and the Half-Blood Prince            |   0.33 |           1 |    1    |         0    |     1    |
| P044_G1 | Remember your name, Chihiro         | Spirited Away                                     |   1    |           1 |    1    |         0    |     1    |
| P046_G1 | My armor was never a distraction... | Iron Man 3                                        |   1    |           0 |    1    |         0    |     1    |
| P047_G1 | They are nice because they’re rich. | Parasite                                          |   1    |           1 |    1    |         0    |     1    |
| P048_G1 | He's not lost. Not anymore.         | Toy Story 4                                       |   0.25 |           0 |    0    |         0    |     0    |
| P049_G1 | The past can hurt. But the way I... | The Lion King                                     |   0    |           0 |    0    |         0.25 |     0.25 |
| P051_G1 | To infinity and beyond              | Toy Story 4                                       |   1    |           0 |    0    |         0    |     1    |
| P052_G1 | elio elio elio oliver oliver oliver | Call Me by Your Name                              |   1    |           0 |    1    |         0.2  |     1    |
| P054_G1 | Once you’ve met someone, you nev... | Spirited Away                                     |   0    |           0 |    0    |         0    |     0    |
| P055_G1 | After all, tomorrow is another day. | Gone with the Wind                                |   1    |           0 |    0    |         0.2  |     1    |
| P056_G1 | Remember who you are, my son.       | The Lion King                                     |   0.5  |           0 |    0    |         0    |     0    |
| P057_G1 | Even the smallest person can cha... | The Lord of the Rings: The Fellowship of the Ring |   1    |           0 |    0    |         0    |     0    |
| P058_G1 | We all go a little mad sometimes.   | Psycho                                            |   1    |           0 |    1    |         0    |     1    |
| P059_G1 | You mustn’t be afraid to dream a... | Inception                                         |   1    |           0 |    0    |         0.25 |     1    |
| P060_G1 | If you run, the beast catches yo... | City Of God                                       |   0.2  |           0 |    0    |         0    |     0    |
| P061_G1 | Nothing has been the same since ... | Iron Man 3                                        |   1    |           0 |    0    |         0    |     0    |

### NHÓM 2: NGỮ NGHĨA & CỐT TRUYỆN (PLOT)
| ID      | Query                               | Target                                            |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   V1_PT2 |
|:--------|:------------------------------------|:--------------------------------------------------|-------:|------------:|--------:|-------------:|---------:|
| P001_G2 | Dad character do anything to get... | Interstellar                                      |   0.25 |         0   |    0    |         0.5  |     0.25 |
| P002_G2 | A innocent prisoner with power t... | The Green Mile                                    |   0    |         0   |    0    |         0    |     0    |
| P004_G2 | A young photographer uses his ca... | City Of God                                       |   1    |         1   |    1    |         0    |     1    |
| P005_G2 | A washed-up former boxer and his... | Real Steel                                        |   1    |         1   |    1    |         0    |     0.5  |
| P007_G2 | A team of dream thieves enters t... | Inception                                         |   1    |         0   |    1    |         0.25 |     1    |
| P009_G2 | The reluctant youngest son of a ... | The Godfather                                     |   0.25 |         0   |    0.5  |         0    |     1    |
| P010_G2 | An alien family faces a brutal, ... | Avatar: Fire and Ash                              |   1    |         0   |    1    |         0    |     1    |
| P011_G2 | A masked vigilante battles a psy... | The Dark Knight                                   |   1    |         1   |    1    |         0    |     1    |
| P012_G2 | A young lion runs away after his... | The Lion King                                     |   0.5  |         0   |    1    |         1    |     1    |
| P013_G2 | A computer programmer discovers ... | The Matrix                                        |   0.33 |         0   |    1    |         0    |     0.25 |
| P014_G2 | A nightclub owner is reunited wi... | Casablanca                                        |   0    |         0   |    0    |         0.2  |     1    |
| P016_G2 | A young wizard discovers an old ... | Harry Potter and the Half-Blood Prince            |   1    |         1   |    1    |         0    |     1    |
| P017_G2 | A wrongfully convicted banker su... | The Shawshank Redemption                          |   0.25 |         0   |    1    |         0    |     1    |
| P018_G2 | A rich southern girl tries to su... | Gone with the Wind                                |   0    |         1   |    0.5  |         1    |     1    |
| P019_G2 | A teenage wizard learns about hi... | Harry Potter and the Half-Blood Prince            |   0.2  |         0   |    1    |         0    |     1    |
| P020_G2 | Twelve jurors locked in a delibe... | 12 Angry Men                                      |   1    |         1   |    1    |         0    |     1    |
| P023_G2 | Determined to protect his family... | The Godfather                                     |   0.33 |         0.5 |    0.5  |         0    |     1    |
| P024_G2 | a poor family tricks their way i... | Parasite                                          |   0    |         1   |    0.5  |         0    |     1    |
| P025_G2 | A reprogrammed cyborg is sent fr... | Terminator 2: Judgment Day                        |   1    |         0   |    0    |         0.33 |     1    |
| P026_G2 | a group of tired soldiers walk a... | Saving Private Ryan                               |   0    |         0   |    1    |         0.25 |     0.5  |
| P027_G2 | A superhero battles a terrorist ... | Iron Man 3                                        |   1    |         0   |    0    |         0    |     0.5  |
| P028_G2 | A computer hacker discovers that... | The Matrix                                        |   1    |         0   |    1    |         0    |     1    |
| P029_G2 | A young lion must overcome loss ... | The Lion King                                     |   1    |         1   |    1    |         1    |     1    |
| P030_G2 | An eccentric inventor builds an ... | Iron Man 3                                        |   0    |         0   |    0    |         0    |     0    |
| P031_G2 | A kind-hearted man with a low IQ... | Forrest Gump                                      |   0.2  |         0   |    0    |         0    |     0    |
| P034_G2 | A skilled thief enters people's ... | Inception                                         |   1    |         0   |    1    |         0.25 |     1    |
| P035_G2 | A loyal cowboy toy embarks on a ... | Toy Story 4                                       |   1    |         0   |    1    |         0.25 |     1    |
| P038_G2 | A simple-minded man sits on a bu... | Forrest Gump                                      |   0.33 |         0   |    0    |         0    |     0    |
| P039_G2 | A lonely desert scavenger discov... | Star Wars: The Force Awakens                      |   0    |         0   |    0    |         0    |     0.5  |
| P043_G2 | A wizard study at Howgarts school   | Harry Potter and the Half-Blood Prince            |   0    |         1   |    1    |         0.33 |     1    |
| P044_G2 | tells the story of a 10-year-old... | Spirited Away                                     |   0.33 |         0   |    1    |         0    |     0.5  |
| P046_G2 | Tony Stark suffers from panic at... | Iron Man 3                                        |   1    |         0   |    1    |         1    |     1    |
| P047_G2 | A poor family gradually infiltra... | Parasite                                          |   0    |         1   |    1    |         0    |     0    |
| P048_G2 | Woody struggles to protect Forky... | Toy Story 4                                       |   1    |         0   |    1    |         0.33 |     1    |
| P049_G2 | A young lion cub must overcome t... | The Lion King                                     |   0    |         0   |    1    |         1    |     1    |
| P051_G2 | A toy helps a new handmade toy f... | Toy Story 4                                       |   1    |         0.5 |    1    |         0.2  |     1    |
| P052_G2 | Two men fall in love, but one of... | Call Me by Your Name                              |   0    |         0   |    0    |         0    |     0    |
| P054_G2 | A young girl enters a spirit wor... | Spirited Away                                     |   0    |         0   |    1    |         0    |     0.25 |
| P055_G2 | A strong-willed woman struggles ... | Gone with the Wind                                |   0    |         0   |    0.25 |         0.5  |     0    |
| P056_G2 | A young lion runs away after los... | The Lion King                                     |   0    |         1   |    1    |         1    |     1    |
| P057_G2 | A young hobbit begins a dangerou... | The Lord of the Rings: The Fellowship of the Ring |   0.5  |         1   |    1    |         0.25 |     1    |
| P058_G2 | A woman stops at a lonely motel ... | Psycho                                            |   0.2  |         0   |    1    |         0    |     1    |
| P059_G2 | A skilled thief enters people’s ... | Inception                                         |   0.5  |         0   |    1    |         1    |     1    |
| P060_G2 | A young boy grows up in a violen... | City Of God                                       |   0.33 |         0   |    1    |         0    |     1    |
| P061_G2 | A brilliant inventor faces a dan... | Iron Man 3                                        |   0    |         0   |    0    |         0    |     0    |

### NHÓM 3: HÌNH ẢNH & BỐI CẢNH (VISUAL SCENE)
| ID      | Query                               | Target                                            |   BM25 |   CLIP_Text |   SBERT |   CLIP_Image |   V1_PT2 |
|:--------|:------------------------------------|:--------------------------------------------------|-------:|------------:|--------:|-------------:|---------:|
| P001_G3 | The tsunami, Deep-Blue, Infinity... | Interstellar                                      |   0    |         0   |    0    |         1    |     1    |
| P002_G3 | Night, Prison                       | The Green Mile                                    |   0    |         0   |    0    |         0    |     0.33 |
| P004_G3 | A chicken frantically dodges cap... | City Of God                                       |   0    |         1   |    1    |         0    |     0    |
| P005_G3 | A boy pulls a glowing, mud-cover... | Real Steel                                        |   1    |         1   |    1    |         0    |     1    |
| P007_G3 | A small metal spinning top twirl... | Inception                                         |   0    |         0   |    0    |         0.5  |     0.33 |
| P010_G3 | sh-covered, blue-skinned warrior... | Avatar: Fire and Ash                              |   1    |         0.5 |    0    |         0    |     0.5  |
| P011_G3 | A caped figure stands brooding o... | The Dark Knight                                   |   0.33 |         1   |    0    |         0    |     0    |
| P012_G3 | An old monkey lifting a baby lio... | The Lion King                                     |   0.25 |         0   |    1    |         0    |     0.25 |
| P013_G3 | A red pill and a blue pill rest ... | The Matrix                                        |   1    |         1   |    0    |         0    |     1    |
| P014_G3 | A smoky black-and-white nightclu... | Casablanca                                        |   1    |         0   |    0    |         0.2  |     1    |
| P016_G3 | A dark magical castle surrounded... | Harry Potter and the Half-Blood Prince            |   1    |         0   |    1    |         0    |     1    |
| P017_G3 | A group of exhausted inmates sit... | The Shawshank Redemption                          |   0.25 |         0   |    0.33 |         0.33 |     1    |
| P018_G3 | A burning city with massive oran... | Gone with the Wind                                |   0    |         0   |    0    |         1    |     1    |
| P019_G3 | An old wizard making a huge wave... | Harry Potter and the Half-Blood Prince            |   0.33 |         1   |    0.5  |         0    |     0.25 |
| P020_G3 | A man pulls out an identical swi... | 12 Angry Men                                      |   0.33 |         1   |    1    |         0    |     1    |
| P023_G3 | Don Vito's dimly lit office, whe... | The Godfather                                     |   1    |         0.5 |    0    |         0    |     0.5  |
| P024_G3 | dirty water flooding a poor base... | Parasite                                          |   1    |         0   |    1    |         0    |     1    |
| P025_G3 | A metal cybernetic arm slowly si... | Terminator 2: Judgment Day                        |   1    |         0   |    1    |         0    |     0.5  |
| P026_G3 | soldiers crawling in bloody wate... | Saving Private Ryan                               |   0    |         1   |    1    |         0.25 |     0.5  |
| P027_G3 | Dozens of armored suits light up... | Iron Man 3                                        |   0    |         0   |    0.25 |         0    |     0.2  |
| P028_G3 | A dark futuristic city filled wi... | The Matrix                                        |   0.2  |         1   |    1    |         0    |     0.33 |
| P029_G3 | A young lion stands alone on a v... | The Lion King                                     |   0.5  |         0   |    1    |         0.5  |     1    |
| P030_G3 | Dozens of glowing, armored suits... | Iron Man 3                                        |   1    |         0.2 |    0    |         0    |     0.25 |
| P031_G3 | A man in a light-colored suit si... | Forrest Gump                                      |   0.2  |         1   |    0    |         0    |     0.33 |
| P034_G3 | A city street folds upward into ... | Inception                                         |   0.5  |         0   |    0    |         0.5  |     0.33 |
| P035_G3 | A massive Ferris wheel glows wit... | Toy Story 4                                       |   0    |         0   |    1    |         0    |     0.25 |
| P038_G3 | A young woman wades through the ... | Forrest Gump                                      |   1    |         1   |    0    |         0    |     0    |
| P039_G3 | A solitary girl slides down a ma... | Star Wars: The Force Awakens                      |   0    |         0   |    0    |         0    |     0    |
| P043_G3 | The old Potions textbook was fil... | Harry Potter and the Half-Blood Prince            |   1    |         1   |    0    |         0.2  |     1    |
| P044_G3 | Chihiro and her parents moved to... | Spirited Away                                     |   1    |         1   |    1    |         0    |     1    |
| P046_G3 | The gathering of the Iron Legion.   | Iron Man 3                                        |   0.5  |         0   |    0    |         0    |     0.5  |
| P047_G3 | A luxurious modern house contras... | Parasite                                          |   0.2  |         1   |    1    |         0    |     0.33 |
| P048_G3 | At a colorful carnival at sunset... | Toy Story 4                                       |   1    |         0   |    1    |         0.25 |     1    |
| P049_G3 | The screen bursts from pitch bla... | The Lion King                                     |   0    |         1   |    1    |         0    |     0    |
| P051_G3 | Colorful toys travel through a c... | Toy Story 4                                       |   0    |         1   |    1    |         0.33 |     1    |
| P052_G3 | Northern Italy — a peaceful coun... | Call Me by Your Name                              |   0    |         0.5 |    0    |         0    |     0    |
| P054_G3 | A magical bathhouse glowing with... | Spirited Away                                     |   0.5  |         0.5 |    1    |         0    |     1    |
| P055_G3 | A grand Southern plantation unde... | Gone with the Wind                                |   0.33 |         0   |    0.5  |         1    |     1    |
| P056_G3 | A vast African savanna glowing u... | The Lion King                                     |   0    |         1   |    1    |         0    |     1    |
| P057_G3 | A peaceful green village surroun... | The Lord of the Rings: The Fellowship of the Ring |   0    |         0   |    0    |         0    |     0    |
| P058_G3 | A dark isolated motel beneath a ... | Psycho                                            |   1    |         0   |    0.5  |         0    |     0.5  |
| P059_G3 | A city street bends upward as bu... | Inception                                         |   0    |         0   |    0    |         0.5  |     0.25 |
| P060_G3 | Narrow streets filled with color... | City Of God                                       |   0    |         1   |    1    |         0    |     0.25 |
| P061_G3 | A high-tech workshop filled with... | Iron Man 3                                        |   1    |         0   |    0.5  |         0    |     0    |

