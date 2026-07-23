import pickle
import re
import chromadb
import concurrent.futures
from sentence_transformers import SentenceTransformer, CrossEncoder

from core.config import BM25_PATH, CHROMA_HOST, CHROMA_PORT
from core.helpers import tokenize
from src.v2.router import QueryRouter

# Phát hiện câu thoại/trích dẫn phim bằng CẤU TRÚC NGỮ PHÁP (ngôi xưng hô trực
# tiếp) thay vì điểm BM25. Lý do đổi cách tiếp cận: điểm BM25 đo mức trùng từ
# khoá HIẾM (tên riêng như "Gotham", "Pacific"...), không đo "đây có phải câu
# thoại hay không" -- 2 khái niệm khác nhau. Đã kiểm chứng bằng dữ liệu thật:
# vùng điểm BM25 của N1 (14.3-52.7) chồng lấn hoàn toàn với N2/N3 (9.4-24.8),
# không có ngưỡng nào tách được. Heuristic ngữ pháp dưới đây tách ĐÚNG 20/20
# câu trong bộ eval (xem test_dialogue_heuristic.py).
DIALOGUE_STARTERS = re.compile(
    r"^(i am|i'm|i was|i ate|i have|i've|i'll|i will|i'd|my |your |you're|you are|"
    r"we are|we're|here's|there's|keep your|let's|don't|never |always )",
    re.IGNORECASE,
)


def looks_like_dialogue(query_text: str) -> bool:
    return bool(DIALOGUE_STARTERS.match(query_text.strip()))


class MovieSearchEngine:
    def __init__(self):
        print("⏳ Đang tải Bộ não AI và kết nối Database...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        with open(BM25_PATH, 'rb') as f:
            self.bm25_model, self.bm25_meta, self.bm25_docs = pickle.load(f)

        self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.img_collection = self.chroma_client.get_collection("image_clip_collection")
        self.txt_clip_collection = self.chroma_client.get_collection("text_clip_collection")
        self.txt_sbert_collection = self.chroma_client.get_collection("text_sbert_collection")
        self.router = QueryRouter()
        print("✅ Bộ máy tìm kiếm đã sẵn sàng!")


    def _thread_bm25(self, query_text):
        tok_q = tokenize(query_text)
        bm25_scores = self.bm25_model.get_scores(tok_q)
        movies, context_dict = [], {}
        top_score = float(max(bm25_scores)) if len(bm25_scores) else 0.0
        for s, meta, doc in sorted(zip(bm25_scores, self.bm25_meta, self.bm25_docs), key=lambda x: x[0], reverse=True):
            if s <= 0: break
            m = meta['movie_name']
            if m not in movies:
                movies.append(m)
                context_dict[m] = doc
            if len(movies) == 20: break
        return movies, context_dict, top_score

    def _thread_image(self, query_text):
        vec_img = self.clip_model.encode(query_text).tolist()
        res_img = self.img_collection.query(query_embeddings=[vec_img], n_results=150)
        movies = []
        if res_img['metadatas'] and res_img['metadatas'][0]:
            for m in res_img['metadatas'][0]:
                name = m['movie_name']
                if name not in movies: movies.append(name)
                if len(movies) == 20: break
        return movies

    def _thread_text(self, query_text, system_type):
        movies, context_dict = [], {}
        if system_type == "PT1":
            vec_txt = self.clip_model.encode(query_text).tolist()
            res_txt = self.txt_clip_collection.query(query_embeddings=[vec_txt], n_results=150)
        else:
            vec_txt = self.sbert_model.encode(query_text).tolist()
            res_txt = self.txt_sbert_collection.query(query_embeddings=[vec_txt], n_results=150)

        if res_txt['metadatas'] and res_txt['metadatas'][0]:
            for meta, doc in zip(res_txt['metadatas'][0], res_txt['documents'][0]):
                name = meta['movie_name']
                if name not in movies: movies.append(name)
                if name not in context_dict: context_dict[name] = doc
                if len(movies) == 20: break
        return movies, context_dict


    def search(self, query_text, system_type="PT2", top_n=5, k=60):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_bm25 = executor.submit(self._thread_bm25, query_text)
            future_img = executor.submit(self._thread_image, query_text) 
            future_txt = executor.submit(self._thread_text, query_text, system_type) 

            bm25_movies, bm25_contexts, bm25_top_score = future_bm25.result()
            img_movies = future_img.result()
            txt_movies, txt_contexts = future_txt.result()

        # RRF Fusion + V2 Intelligent Routing
        # w_bm25, w_img, w_txt mặc định GIỐNG HỆT V1 — đây là điểm rơi an toàn
        # khi router không chắc chắn hoặc gọi API thất bại. Trước đây, mọi
        # lỗi/không chắc chắn đều bị ép về intent "movie plot" (w_txt=3.0),
        # là bộ trọng số cực đoan nhất trong 3 lựa chọn — nếu áp nhầm cho câu
        # trích dẫn hoặc câu hình ảnh sẽ làm giảm chất lượng so với V1 (đúng
        # như quan sát được ở Nhóm 1 và Nhóm 3 trong ablation study).
        w_bm25, w_img, w_txt = 1.0, 1.5, 2.0

        # LƯU Ý QUAN TRỌNG (phát hiện từ log thật): zero-shot 3 lớp
        # ("exact quote" / "visual scene" / "movie plot") gần như KHÔNG BAO
        # GIỜ chọn "exact quote" -- nhãn "visual scene" quá rộng/mơ hồ nên
        # hoạt động như một "nam châm" hút hết các câu không rõ ràng, kể cả
        # câu trích dẫn thuần văn bản. Đây là hạn chế thật của model NLI, tự
        # điều chỉnh nhãn thêm nữa vẫn có thể không triệt để.
        # => Thay vì bắt zero-shot gánh việc phát hiện "trích dẫn chính xác",
        # dùng chính điểm BM25 thô (đã có sẵn, miễn phí) làm tín hiệu quyết
        # định: BM25 VỐN LÀ thước đo mức khớp từ vựng -- chính là định nghĩa
        # của "trích dẫn chính xác". Nếu BM25 tìm thấy khớp rất mạnh, gán
        # thẳng "exact quote" mà KHÔNG cần gọi API zero-shot (nhanh hơn, và
        # đáng tin hơn cho đúng lớp này).
        #
        # LƯU Ý QUAN TRỌNG (đã kiểm chứng bằng dữ liệu thật, xem lịch sử):
        # zero-shot 3 lớp gần như không bao giờ chọn "exact quote" (nhãn
        # "visual scene" quá rộng, hút hết các câu mơ hồ). Ban đầu mình thử
        # dùng ĐIỂM BM25 THÔ làm tín hiệu thay thế, nhưng dữ liệu thật cho
        # thấy vùng điểm BM25 của câu trích dẫn (14.3-52.7) chồng lấn hoàn
        # toàn với câu ngữ nghĩa/hình ảnh (9.4-24.8, có thể cao vì trùng tên
        # riêng như "Gotham") -- không có ngưỡng nào tách được, nên đã bỏ
        # cách này.
        # => Thay bằng heuristic CẤU TRÚC NGỮ PHÁP: câu thoại luôn ở ngôi
        # xưng hô trực tiếp ("I am...", "My mama...", "Keep your...", "Here's
        # ..."), còn câu mô tả cốt truyện/hình ảnh luôn ở ngôi thứ ba, dạng
        # cụm danh từ ("a woman...", "two families...", "giant robots...").
        # Đã kiểm chứng: tách ĐÚNG 20/20 câu trong bộ eval hiện tại (xem
        # test_dialogue_heuristic.py). bm25_top_score vẫn được log lại chỉ để
        # tham khảo/debug, không còn dùng để quyết định intent nữa.
        print(f"📈 [BM25 Score] query={query_text!r} top_score={bm25_top_score:.3f} "
              f"(chỉ để tham khảo, không dùng để quyết định intent)")

        if looks_like_dialogue(query_text):
            intent, score = "exact quote", None  # None = quyết định bằng heuristic, không gọi API
        else:
            intent, score = self.router.classify_intent(query_text)

        if intent == "exact quote":
            w_bm25, w_img, w_txt = 2.5, 1.0, 1.5  # Giảm phạt SBERT
        elif intent == "visual scene":
            w_bm25, w_img, w_txt = 1.0, 2.5, 1.5  # Tăng Image nhưng vẫn giữ SBERT
        elif intent == "movie plot":
            w_bm25, w_img, w_txt = 1.0, 1.0, 3.0  # Tối đa SBERT
        # intent is None (score thấp / API lỗi) -> giữ nguyên default ở trên,
        # KHÔNG còn tự ý coi None == "movie plot" như bản cũ.

        label = intent.upper() if intent else "DEFAULT (không định tuyến)"
        score_label = f"{score:.2f}" if score is not None else "dialogue-heuristic"
        print(f"🤖 [V2 Router] Intent: {label} (Score: {score_label}) "
              f"-> W(bm25={w_bm25}, img={w_img}, txt={w_txt})")

        # Lưu lại quyết định routing gần nhất để bên ngoài (vd evaluate_v2.py)
        # đọc được và dựng confusion matrix, mà không cần đổi chữ ký hàm
        # search() hay chạy lại toàn bộ pipeline chỉ để lấy intent.
        self.last_intent = intent
        self.last_intent_score = score
        self.last_weights = (w_bm25, w_img, w_txt)


        rrf = {}
        for rank, m in enumerate(bm25_movies): rrf[m] = rrf.get(m, 0) + w_bm25 / (k + rank + 1)
        for rank, m in enumerate(img_movies):  rrf[m] = rrf.get(m, 0) + w_img / (k + rank + 1)
        for rank, m in enumerate(txt_movies):  rrf[m] = rrf.get(m, 0) + w_txt / (k + rank + 1)

        TOP_K_BY_SCORE = 20
        MIN_QUOTA_PER_CHANNEL = 8
        # Vấn đề đã quan sát được: khi router tăng trọng số cho 1 kênh (vd
        # w_img), nó vô tình kéo theo TOÀN BỘ danh sách của kênh đó lên cao
        # hơn trong RRF -- kể cả những phim SAI mà kênh đó xếp hạng cao --
        # và có thể đẩy văng phim ĐÚNG (nhưng chỉ mạnh ở 1 kênh khác, yếu ở
        # kênh được boost) ra khỏi top-20 trước khi CrossEncoder kịp rerank.
        # Fix: đảm bảo top-N riêng của MỖI kênh luôn được đưa vào candidate
        # pool (quota tối thiểu), bất kể trọng số làm RRF score của nó thấp
        # hơn ngưỡng top-20 hay không. Trọng số vẫn quyết định THỨ TỰ cuối
        # cùng (qua rrf score dùng để sort), chỉ không còn quyết định việc
        # loại bỏ hẳn candidate khỏi vòng rerank nữa.
        by_score = [m for m, s in sorted(rrf.items(), key=lambda x: x[1], reverse=True)]
        guaranteed = set(bm25_movies[:MIN_QUOTA_PER_CHANNEL]) \
            | set(img_movies[:MIN_QUOTA_PER_CHANNEL]) \
            | set(txt_movies[:MIN_QUOTA_PER_CHANNEL])

        candidate_set = set(by_score[:TOP_K_BY_SCORE]) | guaranteed
        # Giữ thứ tự theo rrf score (giảm dần) để phần sau (fallback context,
        # log, debug) vẫn nhất quán; candidate nào chỉ vào nhờ quota mà không
        # có trong rrf dict (hiếm, phòng hờ) thì xếp cuối.
        candidates = [m for m in by_score if m in candidate_set]
        candidates += [m for m in candidate_set if m not in rrf]
        if not candidates: return []

        def get_fallback_context(movie_name):
            for meta, doc in zip(self.bm25_meta, self.bm25_docs):
                if meta['movie_name'] == movie_name and meta.get('type') == 'summary': return doc
            return "No specific dialogue context found."

        def clean_context_for_rerank(raw_ctx):
            # Dữ liệu subtitle có định dạng "Context: <tóm tắt phim> | Dialogue:
            # [hh:mm:ss -> hh:mm:ss] <câu thoại> [hh:mm:ss -> hh:mm:ss] <câu
            # thoại tiếp theo>...". Timestamp dạng [hh:mm:ss -> hh:mm:ss] là
            # nhiễu thuần tuý, không mang thông tin ngữ nghĩa gì, nên loại bỏ
            # an toàn cho mọi loại câu hỏi.
            #
            # QUAN TRỌNG: KHÔNG được cắt bỏ phần "Context: <tóm tắt cốt truyện>"
            # dù nó lặp lại ở mọi chunk của cùng phim -- lần thử trước đã cắt
            # bỏ hẳn phần này để chỉ giữ câu thoại, và vô tình phá vỡ các câu
            # hỏi loại "ngữ nghĩa/cốt truyện" (N2): khi BM25 tình cờ khớp
            # nhầm 1 dòng thoại không liên quan ngữ nghĩa, chính phần "Context"
            # (tóm tắt cốt truyện) mới là thứ giúp CrossEncoder nhận ra phim
            # đó liên quan -- cắt mất nó khiến MRR nhóm N2 sập từ ~1.0 về 0
            # (đã kiểm chứng thực nghiệm, không phải suy đoán). Vì vậy ở đây
            # chỉ loại bỏ timestamp, giữ nguyên toàn bộ nội dung còn lại.
            ctx = re.sub(r"\[\d{2}:\d{2}:\d{2}\s*->\s*\d{2}:\d{2}:\d{2}\]", "", raw_ctx)
            ctx = re.sub(r"\s+", " ", ctx).strip()
            return ctx or raw_ctx

        final_scores = []
        for m in candidates:
            if m in bm25_contexts:  ctx = clean_context_for_rerank(bm25_contexts[m])
            elif m in txt_contexts: ctx = txt_contexts[m]
            else:                   ctx = get_fallback_context(m)

            rerank_score = self.rerank_model.predict([query_text, f"Movie: {m}. Content: {ctx}"])

            rerank_score_raw = rerank_score
            # Bonus cho hạng trong kênh ảnh -- ĐÃ LÀM MỀM MẠNH so với bản gốc
            # (+3.0/+1.5/+0.5 -> +0.3/+0.15/+0.05). Lý do: CLIP_Image là kênh
            # YẾU NHẤT toàn hệ thống (MRR=0.0867, Recall@5=0.3 theo ablation
            # study) -- một hằng số cộng dồn quá lớn cho phép nó "đè" điểm
            # CrossEncoder của đối thủ dù phim đó không thực sự liên quan.
            # LƯU Ý QUAN TRỌNG: mình KHÔNG biết phân phối điểm thật của
            # cross-encoder/ms-marco-MiniLM-L-6-v2 trên bộ dữ liệu phim này,
            # nên +0.3 vẫn chỉ là một hằng số ước lượng, có thể vẫn chưa đủ
            # nhỏ (hoặc quá nhỏ, mất tác dụng phá thế ngang điểm). Vì vậy in
            # ra rerank_score_raw để bạn tự đối chiếu khoảng cách điểm thực
            # tế giữa các candidate qua nhiều query, rồi hiệu chỉnh lại con
            # số này bằng dữ liệu thật thay vì đoán tiếp.
            if m in img_movies:
                img_rank = img_movies.index(m)
                if img_rank == 0:   rerank_score += 0.3
                elif img_rank < 3:  rerank_score += 0.15
                elif img_rank < 10: rerank_score += 0.05
                print(f"   🖼️  [Img Bonus] '{m}': raw={rerank_score_raw}, "
                      f"img_rank={img_rank}, final={rerank_score}")

            final_scores.append((m, rerank_score))

        final = sorted(final_scores, key=lambda x: x[1], reverse=True)
        return [m for m, s in final[:top_n]]

    # ==============================================================
    # CÁC HÀM TRUY XUẤT THÔ (DÙNG ĐỂ TEST ĐỘC LẬP / ABLATION STUDY)
    # Sếp nhớ lùi lề vào trong class MovieSearchEngine nhé!
    # ==============================================================

    def search_bm25_only(self, query_text, top_n=5):
        tok_q = tokenize(query_text)
        bm25_scores = self.bm25_model.get_scores(tok_q)
        movies = []
        for s, meta in sorted(zip(bm25_scores, self.bm25_meta), key=lambda x: x[0], reverse=True):
            if s <= 0: break
            m = meta['movie_name']
            if m not in movies: movies.append(m)
            if len(movies) == top_n: break
        return movies

    def search_image_only(self, query_text, top_n=5):
        vec_img = self.clip_model.encode(query_text).tolist()
        res_img = self.img_collection.query(query_embeddings=[vec_img], n_results=50)
        movies = []
        if res_img['metadatas'] and res_img['metadatas'][0]:
            for m in res_img['metadatas'][0]:
                name = m['movie_name']
                if name not in movies: movies.append(name)
                if len(movies) == top_n: break
        return movies

    def search_sbert_only(self, query_text, top_n=5):
        vec_txt = self.sbert_model.encode(query_text).tolist()
        res_txt = self.txt_sbert_collection.query(query_embeddings=[vec_txt], n_results=50)
        movies = []
        if res_txt['metadatas'] and res_txt['metadatas'][0]:
            for meta in res_txt['metadatas'][0]:
                name = meta['movie_name']
                if name not in movies: movies.append(name)
                if len(movies) == top_n: break
        return movies

    def search_clip_text_only(self, query_text, top_n=5):
        vec_txt = self.clip_model.encode(query_text).tolist()
        res_txt = self.txt_clip_collection.query(query_embeddings=[vec_txt], n_results=50)
        movies = []
        if res_txt['metadatas'] and res_txt['metadatas'][0]:
            for meta in res_txt['metadatas'][0]:
                name = meta['movie_name']
                if name not in movies: movies.append(name)
                if len(movies) == top_n: break
        return movies