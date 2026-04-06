import pandas as pd
from db_builder import DatabaseBuilder
from search_engine import MovieSearchEngine


def show_table(title, columns_data):
    """Hàm in bảng kết quả siêu đẹp"""
    print(f"\n{'=' * 80}")
    print(f"🎯 {title}")
    print(f"{'=' * 80}")
    df = pd.DataFrame(columns_data)
    print(df.to_markdown(index=False))


def run_app():
    while True:
        print("\n" + "🌟" * 20)
        print("🎬 BẢNG ĐIỀU KHIỂN HỆ THỐNG TRUY XUẤT PHIM")
        print("🌟" * 20)
        print("1. 🛠️ Khởi tạo/Nạp lại Database")
        print("2. 🚀 Tìm kiếm TỔNG HỢP (Hệ thống hoàn chỉnh RRF + Rerank)")
        print("3. 🧠 So sánh SBERT vs CLIP Text (Test Ngữ nghĩa)")
        print("4. 🔑 Tìm kiếm thuần BM25 (Test Từ khóa/Quote)")
        print("5. 👁️ Tìm kiếm thuần CLIP Image (Test tìm cảnh phim)")
        print("6. ❌ Thoát")

        choice = input("👉 Chọn chế độ test (1-6): ").strip()

        if choice == '1':
            builder = DatabaseBuilder()
            builder.execute()

        elif choice in ['2', '3', '4', '5']:
            print("⏳ Đang khởi động AI Engine...")
            engine = MovieSearchEngine()

            while True:
                user_q = input("\n🔍 Gõ câu tìm kiếm (hoặc gõ '0' để quay lại Menu): ").strip()
                if user_q == '0': break
                if not user_q: continue

                # CHẾ ĐỘ 2: TỔNG HỢP
                if choice == '2':
                    pt1 = engine.search(user_q, system_type="PT1", top_n=5)
                    pt2 = engine.search(user_q, system_type="PT2", top_n=5)
                    show_table(f"TỔNG HỢP: '{user_q.upper()}'", {
                        "Hạng": [f"Top {i + 1}" for i in range(5)],
                        "PT1 (Clip text + image, bm25)": pt1 + ["-"] * (5 - len(pt1)),
                        "PT2 (Clip image, sbert, bm25)": pt2 + ["-"] * (5 - len(pt2))
                    })

                # CHẾ ĐỘ 3: ĐẤU TRƯỜNG NGỮ NGHĨA SBERT VÀ CLIP TEXT
                elif choice == '3':
                    sbert_res = engine.search_sbert_only(user_q, top_n=5)
                    clip_res = engine.search_clip_text_only(user_q, top_n=5)
                    show_table(f"SBERT vs CLIP TEXT: '{user_q.upper()}'", {
                        "Hạng": [f"Top {i + 1}" for i in range(5)],
                        "SBERT (Chuyên gia Ngữ nghĩa)": sbert_res + ["-"] * (5 - len(sbert_res)),
                        "CLIP Text (Dịch chữ ra hình)": clip_res + ["-"] * (5 - len(clip_res))
                    })

                # CHẾ ĐỘ 4: CHỈ CHẠY TỪ KHÓA BM25
                elif choice == '4':
                    bm25_res = engine.search_bm25_only(user_q, top_n=5)
                    show_table(f"BM25 KEYWORD: '{user_q.upper()}'", {
                        "Hạng": [f"Top {i + 1}" for i in range(5)],
                        "Phim tìm được": bm25_res + ["-"] * (5 - len(bm25_res))
                    })

                # CHẾ ĐỘ 5: CHỈ TÌM ẢNH BẰNG CLIP IMAGE
                elif choice == '5':
                    img_res = engine.search_image_only(user_q, top_n=5)
                    show_table(f"CLIP IMAGE SEARCH: '{user_q.upper()}'", {
                        "Hạng": [f"Top {i + 1}" for i in range(5)],
                        "Phim có cảnh khớp nhất": img_res + ["-"] * (5 - len(img_res))
                    })

        elif choice == '6':
            print("👋 Tạm biệt sếp! Chúc sếp bảo vệ đồ án điểm tuyệt đối!")
            break
        else:
            print("⚠️ Lựa chọn không hợp lệ!")


if __name__ == "__main__":
    run_app()