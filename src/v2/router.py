import os
from huggingface_hub import InferenceClient

class QueryRouter:
    def __init__(self, model_id="facebook/bart-large-mnli"):
        print("⏳ Khởi tạo AI Router (Hugging Face API)...")
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("⚠️ CẢNH BÁO: Chưa cấu hình HF_TOKEN trong biến môi trường. Gọi API có thể bị giới hạn hoặc lỗi 401.")
        
        self.client = InferenceClient(model_id, token=token)
        # Dùng các từ vựng nguyên thủy nhất để tránh mô hình bị bias (thiên lệch)
        self.label_mapping = {
            "dialogue": "exact quote",
            "story": "movie plot",
            "picture": "visual scene"
        }
        self.candidate_labels = list(self.label_mapping.keys())

    def classify_intent(self, query_text):
        """
        Gửi query lên HuggingFace Inference API để phân loại zero-shot.
        Trả về tuple (top_intent, confidence_score)
        """
        try:
            # Tham số cho zero_shot_classification: text, candidate_labels
            result = self.client.zero_shot_classification(text=query_text, candidate_labels=self.candidate_labels)
            
            # Khác với pipeline trả về dict {'labels': [...], 'scores': [...]}, 
            # object trả về của API InferenceClient (ví dụ: list of dicts hoặc object có thuộc tính)
            
            # API InferenceClient zero_shot_classification có thể trả về một object dạng dict
            # Kiểm tra dictionary properties
            if hasattr(result, 'labels') and hasattr(result, 'scores'):
                top_intent_raw = result.labels[0]
                score = result.scores[0]
            elif isinstance(result, list) and len(result) > 0 and 'label' in result[0]:
                top_intent_raw = result[0]['label']
                score = result[0]['score']
            elif isinstance(result, dict) and 'labels' in result:
                top_intent_raw = result['labels'][0]
                score = result['scores'][0]
            else:
                # print(f"⚠️ Format kết quả: {result}")
                # Hỗ trợ thuộc tính trực tiếp nếu nó là parse object từ huggingface_hub
                try:
                    top_intent_raw = result[0].label if hasattr(result[0], 'label') else result.labels[0]
                    score = result[0].score if hasattr(result[0], 'score') else result.scores[0]
                except:
                    top_intent_raw = "a summary of the movie's plot or meaning" # fallback
                    score = 0.33
                
            # Ánh xạ từ nhãn dài về nhãn ngắn cho hệ thống search engine
            top_intent = self.label_mapping.get(top_intent_raw, "movie plot")
            return top_intent, score
        except Exception as e:
            print(f"⚠️ Lỗi gọi HF API: {e}. Fallback về mặc định (movie plot).")
            return "movie plot", 0.0

if __name__ == "__main__":
    router = QueryRouter()
    queries = [
        "a glowing mechanical suit flying in the sky",
        "I am going to make him an offer he can't refuse",
        "a guy with short term memory loss taking polaroid pictures"
    ]
    for q in queries:
        intent, score = router.classify_intent(q)
        print(f"Query: '{q}' -> Intent: {intent} (Score: {score:.4f})")
