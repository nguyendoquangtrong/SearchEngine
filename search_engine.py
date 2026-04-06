import pickle
import chromadb
import concurrent.futures
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import BM25_PATH, CHROMA_HOST, CHROMA_PORT
from helpers import tokenize

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
        print("✅ Bộ máy tìm kiếm đã sẵn sàng!")

    def _thread_bm25(self, query_text):
        tok_q = tokenize(query_text)
        bm25_scores = self.bm25_model.get_scores(tok_q)
        movies, context_dict = [], {}
        for s, meta, doc in sorted(zip(bm25_scores, self.bm25_meta, self.bm25_docs), key=lambda x: x[0], reverse=True):
            if s <= 0: break
            m = meta['movie_name']
            if m not in movies:
                movies.append(m)
                context_dict[m] = doc
            if len(movies) == 20: break
        return movies, context_dict

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

            bm25_movies, bm25_contexts = future_bm25.result()
            img_movies = future_img.result()
            txt_movies, txt_contexts = future_txt.result()

        # RRF Fusion
        rrf = {}
        for rank, m in enumerate(bm25_movies): rrf[m] = rrf.get(m, 0) + 1.0 / (k + rank + 1)
        for rank, m in enumerate(img_movies):  rrf[m] = rrf.get(m, 0) + 1.5 / (k + rank + 1)
        for rank, m in enumerate(txt_movies):  rrf[m] = rrf.get(m, 0) + 2.0 / (k + rank + 1)
            
        candidates = [m for m, s in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:20]]
        if not candidates: return []

        def get_fallback_context(movie_name):
            for meta, doc in zip(self.bm25_meta, self.bm25_docs):
                if meta['movie_name'] == movie_name and meta.get('type') == 'summary': return doc
            return "No specific dialogue context found."

        final_scores = []
        for m in candidates:
            if m in bm25_contexts:  ctx = bm25_contexts[m]
            elif m in txt_contexts: ctx = txt_contexts[m]
            else:                   ctx = get_fallback_context(m)

            rerank_score = self.rerank_model.predict([query_text, f"Movie: {m}. Content: {ctx}"])

            if m in img_movies:
                img_rank = img_movies.index(m)
                if img_rank == 0:   rerank_score += 3.0
                elif img_rank < 3:  rerank_score += 1.5
                elif img_rank < 10: rerank_score += 0.5

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