import pandas as pd
from search_engine import MovieSearchEngine


class AblationEvaluator:
    def __init__(self):
        print("⏳ Đang khởi động AI Engine cho Ablation Study (5 Models)...")
        self.engine = MovieSearchEngine()

        # ==========================================
        # 🧪 BỘ TEST CASE ĐÃ THÊM NHÃN "group"
        # ==========================================
        self.test_cases = [
            # ----------------------------------------------------------------
            # NHÓM 1: TRÍCH DẪN CHÍNH XÁC (Khảo nghiệm sức mạnh BM25)
            # ----------------------------------------------------------------
            {"group": "Nhóm 1: Trích dẫn chính xác", "query": "I am going to make him an offer he can't refuse",
             "expected": ["The Godfather"]},
            {"group": "Nhóm 1: Trích dẫn chính xác", "query": "My mama always said life was like a box of chocolates",
             "expected": ["Forrest Gump"]},
            {"group": "Nhóm 1: Trích dẫn chính xác", "query": "I ate his liver with some fava beans and a nice Chianti",
             "expected": ["The Silence of the Lambs"]},
            {"group": "Nhóm 1: Trích dẫn chính xác", "query": "Keep your friends close, but your enemies closer",
             "expected": ["The Godfather Part III"]},
            {"group": "Nhóm 1: Trích dẫn chính xác", "query": "Here's looking at you, kid", "expected": ["Casablanca"]},

            # ----------------------------------------------------------------
            # NHÓM 2: TÌM THEO NGỮ NGHĨA / CỐT TRUYỆN (Sân khấu của SBERT)
            # ----------------------------------------------------------------
            {"group": "Nhóm 2: Ngữ nghĩa & Cốt truyện",
             "query": "two completely opposite families one extremely rich and the other always lives in poverty",
             "expected": ["Parasite"]},
            {"group": "Nhóm 2: Ngữ nghĩa & Cốt truyện",
             "query": "a computer hacker learning the truth about his simulated reality", "expected": ["The Matrix"]},
            {"group": "Nhóm 2: Ngữ nghĩa & Cốt truyện", "query": "a banker wrongly convicted of murder escapes prison",
             "expected": ["The Shawshank Redemption"]},
            {"group": "Nhóm 2: Ngữ nghĩa & Cốt truyện", "query": "entering dreams to steal information from a target",
             "expected": ["Inception"]},
            {"group": "Nhóm 2: Ngữ nghĩa & Cốt truyện", "query": "two gangsters, a boxer, and a stolen briefcase",
             "expected": ["Pulp Fiction"]},
            {"group": "Nhóm 2: Ngữ nghĩa & Cốt truyện",
             "query": "a girl trying to save her parents who turned into pigs", "expected": ["Spirited Away"]},
            {"group": "Nhóm 2: Ngữ nghĩa & Cốt truyện",
             "query": "brother and sister struggling to survive during World War II in Japan",
             "expected": ["Grave Of The Fireflies"]},

            # ----------------------------------------------------------------
            # NHÓM 3: TÌM THEO THỊ GIÁC / BỐI CẢNH (Quyền năng của CLIP Image)
            # ----------------------------------------------------------------
            {"group": "Nhóm 3: Hình ảnh & Bối cảnh", "query": "a woman screaming in a motel shower black and white",
             "expected": ["Psycho"]},
            {"group": "Nhóm 3: Hình ảnh & Bối cảnh", "query": "a glowing mechanical suit flying in the sky",
             "expected": ["Iron Man 3"]},
            {"group": "Nhóm 3: Hình ảnh & Bối cảnh", "query": "giant robots fighting monsters in the ocean",
             "expected": ["Pacific Rim: Uprising"]},
            {"group": "Nhóm 3: Hình ảnh & Bối cảnh", "query": "seven warriors defending a village in the rain",
             "expected": ["Seven Samurai"]},
            {"group": "Nhóm 3: Hình ảnh & Bối cảnh", "query": "a dark knight standing on a tall building in Gotham",
             "expected": ["The Dark Knight"]},

            # ----------------------------------------------------------------
            # NHÓM 4: BẪY TỪ VỰNG / TÊN RIÊNG / SAI LỆCH (Test độ lì của RRF)
            # ----------------------------------------------------------------
            {"group": "Nhóm 4: Bẫy từ vựng & Sai lệch",
             "query": "a guy with short term memory loss taking polaroid pictures", "expected": ["Memento"]},
            {"group": "Nhóm 4: Bẫy từ vựng & Sai lệch", "query": "two magicians competing and sabotaging each other",
             "expected": ["The Prestige"]},
            {"group": "Nhóm 4: Bẫy từ vựng & Sai lệch", "query": "I am going to make him an offer he cannot refuse",
             "expected": ["The Godfather"]}
        ]

    # ==========================================
    # CÁC HÀM TÍNH ĐIỂM (EVALUATION METRICS)
    # ==========================================
    def get_mrr(self, predicted, expected):
        """Tính chỉ số Mean Reciprocal Rank"""
        for i, p in enumerate(predicted):
            if p in expected:
                return 1.0 / (i + 1)
        return 0.0

    def get_precision_at_k(self, predicted, expected, k=5):
        """Tính Precision@K"""
        top_k = predicted[:k]
        hits = sum(1 for p in top_k if p in expected)
        return hits / k if k > 0 else 0.0

    def get_recall_at_k(self, predicted, expected, k=5):
        """Tính Recall@K"""
        top_k = predicted[:k]
        hits = sum(1 for p in top_k if p in expected)
        return hits / len(expected) if expected else 0.0

    # ==========================================
    # CHẠY ĐÁNH GIÁ
    # ==========================================
    def run_ablation_study(self, top_n=5):
        print(f"\n🚀 ĐANG CHẠY ABLATION STUDY (So sánh 5 luồng - MRR, Precision@{top_n}, Recall@{top_n})...")
        print("-" * 110)

        results = []

        # Khởi tạo Dictionary theo dõi cả 3 chỉ số cho 5 hệ thống
        models = ["BM25_Only", "CLIP_Text_Only", "SBERT_Only", "CLIP_Image_Only", "Multimodal_PT2"]
        metrics = {model: {"mrr": 0.0, "p_at_k": 0.0, "r_at_k": 0.0} for model in models}

        for idx, test in enumerate(self.test_cases):
            query = test["query"]
            expected = test["expected"]

            # Chạy 5 luồng độc lập
            preds = {
                "BM25_Only": self.engine.search_bm25_only(query, top_n=top_n),
                "CLIP_Text_Only": self.engine.search_clip_text_only(query, top_n=top_n),
                "SBERT_Only": self.engine.search_sbert_only(query, top_n=top_n),
                "CLIP_Image_Only": self.engine.search_image_only(query, top_n=top_n),
                "Multimodal_PT2": self.engine.search(query, system_type="PT2", top_n=top_n)
            }

            # Tính và cộng dồn điểm cho từng hệ thống
            row_result = {
                "Phân Vùng": test["group"],
                "Query": query[:30] + "..."
            }

            for model_name, pred_list in preds.items():
                mrr = self.get_mrr(pred_list, expected)
                p_at_k = self.get_precision_at_k(pred_list, expected, k=top_n)
                r_at_k = self.get_recall_at_k(pred_list, expected, k=top_n)

                metrics[model_name]["mrr"] += mrr
                metrics[model_name]["p_at_k"] += p_at_k
                metrics[model_name]["r_at_k"] += r_at_k

                # Chỉ lưu MRR vào bảng chi tiết để tránh bảng quá to gãy giao diện
                short_name = model_name.replace("_Only", "").replace("Multimodal_", "")
                row_result[short_name] = round(mrr, 2)

            results.append(row_result)
            print(f"[{idx + 1:02d}/{len(self.test_cases)}] Đã test xong: '{query[:30]}...'")

        # In Bảng so sánh chi tiết (Theo MRR)
        df = pd.DataFrame(results)
        print("\n" + "=" * 110)
        print("📊 BẢNG SO SÁNH MRR TỪNG CÂU QUERY (CHI TIẾT)")
        print("=" * 110)

        grouped = df.groupby("Phân Vùng", sort=False)
        for name, group in grouped:
            print(f"\n📌 {name.upper()}")
            print(group.drop(columns=["Phân Vùng"]).to_markdown(index=False))

        # ==========================================
        # In Tổng kết điểm 3 Chiều (MRR, Precision, Recall)
        # ==========================================
        n = len(self.test_cases)
        print("\n" + "=" * 110)
        print(f"🏆 TỔNG KẾT ĐIỂM SỐ ĐA CHIỀU (TRUNG BÌNH {n} CÂU TRUY VẤN)")
        print("=" * 110)

        # Tạo bảng tóm tắt tổng
        summary_data = []
        for model in models:
            summary_data.append({
                "Luồng Mô Hình": model.replace("_Only", "").replace("Multimodal_", "🚀 "),
                "MRR": f"{metrics[model]['mrr'] / n:.4f}",
                f"Precision@{top_n}": f"{metrics[model]['p_at_k'] / n:.4f}",
                f"Recall@{top_n}": f"{metrics[model]['r_at_k'] / n:.4f}"
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_markdown(index=False))
        print("=" * 110 + "\n")


if __name__ == "__main__":
    evaluator = AblationEvaluator()
    evaluator.run_ablation_study(top_n=5)