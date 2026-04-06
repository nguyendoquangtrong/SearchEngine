import os
import math
import json
import pickle
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from deep_translator import GoogleTranslator

# Gọi đồ nghề từ 2 file trước
from config import *
from helpers import make_folder_name, normalize_name, tokenize

class DatabaseBuilder:
    def __init__(self):
        self.translator = GoogleTranslator(source='vi', target='en')
        self.clip_model = None
        self.sbert_model = None

    def clean_and_translate(self):
        print("\n" + "="*50)
        print("⏳ BƯỚC 1: DỌN DẸP & DỊCH THUẬT DỮ LIỆU...")
        print("="*50)
        
        existing_folders = set(os.listdir(MOVIE_FOLDERS)) if os.path.exists(MOVIE_FOLDERS) else set()
        
        with open(RAW_JSON_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        cleaned_data = []
        seen_titles = set()
        total_raw = len(raw_data)

        for idx, item in enumerate(raw_data):
            title_goc = str(item.get('origin_name', item.get('title', ''))).strip()
            title_viet = str(item.get('title', '')).strip()
            title_lower = title_goc.lower()

            if not title_goc or title_lower in seen_titles: continue
            if str(item.get('type', '')).lower() == 'series' or (str(item.get('episode_total', '1')).isdigit() and int(item.get('episode_total', '1')) > 2): continue

            folder_1, folder_2 = make_folder_name(title_goc), make_folder_name(title_viet)
            if not any(name in existing_folders for name in [title_goc, title_viet, folder_1, folder_2]): 
                continue

            print(f"  [{idx+1:03d}/{total_raw}] Đang map & dịch: '{title_goc}'...", end=" ")

            try:
                vi_content = str(item.get('content', '')).strip()
                item['content'] = self.translator.translate(vi_content[:4900]) if vi_content else "No content available."
                item['title'] = title_goc
                cleaned_data.append(item)
                seen_titles.add(title_lower)
                print("✅ OK")
            except Exception as e:
                print(f"⚠️ Lỗi: {e}")

        with open(CLEAN_EN_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
        
        print(f"\n🎉 TỔNG KẾT BƯỚC 1: Đã dọn xong {len(cleaned_data)} phim hợp lệ!")
        return cleaned_data

    def build_vector_db(self, cleaned_data):
        print("\n" + "="*50)
        print("⏳ BƯỚC 2: TẢI MÔ HÌNH VÀ KHỞI TẠO CHROMADB...")
        print("="*50)
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        for col in ["image_clip_collection", "text_clip_collection", "text_sbert_collection"]:
            try: chroma_client.delete_collection(name=col)
            except: pass
        
        img_collection = chroma_client.create_collection("image_clip_collection")
        txt_clip_collection = chroma_client.create_collection("text_clip_collection")
        txt_sbert_collection = chroma_client.create_collection("text_sbert_collection")

        print("\n" + "="*50)
        print("⏳ BƯỚC 3: XỬ LÝ TEXT VÀ HÌNH ẢNH (SLIDING WINDOW)...")
        print("="*50)
        docs, bm25_docs, txt_metas, txt_ids = [], [], [], []
        folder_map = {normalize_name(f): f for f in os.listdir(MOVIE_FOLDERS)}
        
        total_clean = len(cleaned_data)

        for i, item in enumerate(cleaned_data):
            title_goc = item.get('title', 'Unknown')
            norm_title = normalize_name(title_goc)
            global_desc = str(item.get('content', 'No content')).strip()

            docs.append(f"Movie Summary: {global_desc}")
            txt_metas.append({"movie_name": title_goc, "type": "summary"})
            txt_ids.append(f"txt_{norm_title}_summary")

            target_folder = folder_map.get(norm_title)
            script_found = False

            if target_folder:
                script_folder = os.path.join(MOVIE_FOLDERS, target_folder, 'script')
                if os.path.exists(script_folder):
                    for file_name in os.listdir(script_folder):
                        if file_name.endswith('.txt'):
                            try:
                                with open(os.path.join(script_folder, file_name), 'r', encoding='utf-8') as sf:
                                    lines = sf.readlines()
                                
                                valid_dialogues = [line.strip() for line in lines if len((line.split(']', 1)[1].replace(':', '', 1).strip() if len(line.split(']', 1)) > 1 else line.strip()).split()) >= 3]
                                
                                # =========================================================
                                # 🎬 THUẬT TOÁN CHUNKING: CỬA SỔ TRƯỢT (SLIDING WINDOW)
                                # =========================================================
                                WINDOW_SIZE = 4  # Lấy 4 câu thoại
                                STEP = 2         # Trượt 2 câu (Overlap 2 câu)

                                for j in range(0, len(valid_dialogues), STEP):
                                    chunk_lines = valid_dialogues[j:j+WINDOW_SIZE]
                                    if not chunk_lines: break
                                    
                                    chunk_text = " ".join(chunk_lines)
                                    short_desc = global_desc[:100] + "..." if len(global_desc) > 100 else global_desc
                                    
                                    docs.append(f"Context: {short_desc} | Dialogue: {chunk_text}")        
                                    bm25_docs.append(chunk_text)  
                                    txt_metas.append({"movie_name": title_goc, "type": "subtitle"})
                                    txt_ids.append(f"txt_{norm_title}_sub_chunk_{j}")
                                # =========================================================
                                script_found = True
                            except Exception as e: 
                                print(f"    ⚠️ Lỗi đọc script {title_goc}: {e}")
                            break 
            
            status = "✅ Có kịch bản" if script_found else "⚠️ Chỉ có tóm tắt"
            print(f"  [{i+1:03d}/{total_clean}] Lọc dữ liệu Text: '{title_goc}' -> {status}")

        print(f"\n📚 Đang lập chỉ mục từ khóa (BM25) cho {len(bm25_docs)} đoạn hội thoại...")
        bm25_model = BM25Okapi([tokenize(doc) for doc in bm25_docs])
        with open(BM25_PATH, 'wb') as f: pickle.dump((bm25_model, txt_metas, docs), f)
        print("✅ Lưu BM25 thành công!")

        print("\n🖼️ Đang quét đường dẫn Ảnh...")
        image_paths, img_metas, img_ids = [], [], []
        valid_ext = ('.jpg', '.jpeg', '.png', '.webp')
        
        for i, item in enumerate(cleaned_data):
            title_goc = item.get('title', 'Unknown')
            target_folder = folder_map.get(normalize_name(title_goc))
            img_count = 0
            
            if target_folder:
                pic_folder = os.path.join(MOVIE_FOLDERS, target_folder, 'picture')
                if os.path.exists(pic_folder):
                    for f_name in os.listdir(pic_folder):
                        f_path = os.path.join(pic_folder, f_name)
                        if os.path.isfile(f_path) and f_name.lower().endswith(valid_ext):
                            image_paths.append(f_path)
                            img_metas.append({"movie_name": title_goc, "file_name": f_name})
                            img_ids.append(f"img_{normalize_name(title_goc)}_{len(image_paths)}")
                            img_count += 1
            
            print(f"  [{i+1:03d}/{total_clean}] Quét ảnh: '{title_goc}' -> Tìm thấy {img_count} frames")

        BATCH_SIZE = 32
        print("\n" + "="*50)
        print("🚀 BƯỚC 4: ĐANG NÉN VECTOR VÀO KHO (Sẽ mất thời gian)...")
        print("="*50)
        
        print(f"🧠 Đang nén SBERT cho {len(docs)} đoạn Text...")
        sbert_vecs = self.sbert_model.encode(docs, batch_size=BATCH_SIZE, show_progress_bar=True).tolist()
        
        print(f"👁️ Đang nén CLIP cho {len(docs)} đoạn Text...")
        txt_clip_vecs = self.clip_model.encode(docs, batch_size=BATCH_SIZE, show_progress_bar=True).tolist()

        print("💾 Đang đẩy Text Vector vào Database Docker...")
        for i in range(0, len(docs), BATCH_SIZE):
            txt_sbert_collection.add(embeddings=sbert_vecs[i:i+BATCH_SIZE], documents=docs[i:i+BATCH_SIZE], metadatas=txt_metas[i:i+BATCH_SIZE], ids=txt_ids[i:i+BATCH_SIZE])
            txt_clip_collection.add(embeddings=txt_clip_vecs[i:i+BATCH_SIZE], documents=docs[i:i+BATCH_SIZE], metadatas=txt_metas[i:i+BATCH_SIZE], ids=txt_ids[i:i+BATCH_SIZE])

        print(f"\n🖼️ Đang nén CLIP cho {len(image_paths)} Hình ảnh...")
        total_img_batches = math.ceil(len(image_paths) / BATCH_SIZE)
        
        for i in range(0, len(image_paths), BATCH_SIZE):
            b_paths = image_paths[i:i+BATCH_SIZE]
            b_images = [Image.open(p).convert("RGB") if os.path.exists(p) else Image.new('RGB', (224, 224), color='black') for p in b_paths]
            b_vecs = self.clip_model.encode(b_images, show_progress_bar=False).tolist()
            img_collection.add(embeddings=b_vecs, metadatas=img_metas[i:i+BATCH_SIZE], ids=img_ids[i:i+BATCH_SIZE])
            
            if (i // BATCH_SIZE + 1) % 5 == 0 or (i // BATCH_SIZE + 1) == total_img_batches: 
                print(f"  -> Đã lưu batch ảnh {i // BATCH_SIZE + 1}/{total_img_batches} vào Database...")

        print("\n🎉 HOÀN TẤT! DATABASE ĐÃ ĐƯỢC XÂY DỰNG BẰNG SLIDING WINDOW XONG!")

    def execute(self):
        cleaned_data = self.clean_and_translate()
        self.build_vector_db(cleaned_data)