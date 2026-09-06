import os
import glob
import re
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def clean_and_validate_caption(raw_text: str) -> str:
    """Làm sạch và kiểm định chất lượng caption đầu ra từ VLM."""
    text = raw_text.strip()

    # Cắt bỏ nếu AI bị lặp lại cụm mồi "The scene shows"
    if text.lower().startswith("the scene shows"):
        text = text[15:].strip()

    # Loại bỏ các tiền tố lửng lơ vô nghĩa
    text = re.sub(r"^(is|shows|depicts|showing)\s+", "", text, flags=re.IGNORECASE)

    # Loại bỏ rác ký tự lặp
    text = re.sub(r"[_\.\-=\*|]{2,}", "", text).strip()

    # Lọc ngôn ngữ lạ (Chỉ chấp nhận ký tự chuẩn ASCII Tiếng Anh)
    if not all(32 <= ord(char) <= 126 for char in text):
        return ""

    # Loại bỏ các câu quá ngắn
    if len(text) < 10 or text.lower() in ["none", "nothing", "the scene"]:
        return ""

    # Gắn lại cụm từ mồi để tạo thành một đoạn văn hoàn chỉnh
    if text:
        text = "The scene shows " + text[0].lower() + text[1:]

    return text


class VLMCaptionGenerator:
    """Class chịu trách nhiệm sinh Caption từ ảnh bằng mô hình VLM (BLIP-2)."""

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = None):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Thư viện 'transformers' chưa được cài đặt. "
                "Vui lòng cài đặt qua: pip install transformers accelerate"
            )

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.processor = None
        self.model = None

    def load_model(self):
        """Nạp mô hình BLIP-2 vào bộ nhớ GPU/CPU."""
        if self.model is not None:
            return

        print(f"[*] Đang chạy sinh caption trên thiết bị: {self.device.upper()}")
        print(f"[*] Đang nạp BLIP-2 Model ({self.model_name})...")

        self.processor = Blip2Processor.from_pretrained(self.model_name)
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype
        ).to(self.device)

        print("[+] Nạp mô hình VLM thành công!\n")

    def process_movie(
        self,
        movie_path: str,
        movie_name: str,
        batch_size: int = 16,
        overwrite: bool = False
    ) -> str:
        """Sinh caption cho tất cả các hình ảnh thuộc một bộ phim.

        Args:
            movie_path: Đường dẫn thư mục phim chứa thư mục 'picture'.
            movie_name: Tên phim.
            batch_size: Kích thước batch khi suy luận.
            overwrite: Nếu True, ghi đè file caption hiện có.

        Returns:
            Đường dẫn file caption đã được tạo/cập nhật.
        """
        picture_dir = os.path.join(movie_path, "picture")
        output_file = os.path.join(movie_path, f"{movie_name}_captions.txt")

        if not os.path.exists(picture_dir):
            return output_file

        image_paths = sorted(
            glob.glob(os.path.join(picture_dir, "*.jpg")) +
            glob.glob(os.path.join(picture_dir, "*.jpeg")) +
            glob.glob(os.path.join(picture_dir, "*.png")) +
            glob.glob(os.path.join(picture_dir, "*.webp"))
        )

        if not image_paths:
            return output_file

        if os.path.exists(output_file) and not overwrite:
            print(f"  ℹ️ File caption đã tồn tại: {output_file} (Bỏ qua VLM)")
            return output_file

        self.load_model()

        print(f"[*] Đang xử lý sinh caption cho phim: {movie_name} ({len(image_paths)} ảnh)")

        with open(output_file, "w", encoding="utf-8") as f:
            for i in tqdm(range(0, len(image_paths), batch_size), desc=f"  Captions {movie_name}"):
                batch_paths = image_paths[i : i + batch_size]
                images = []
                valid_paths = []

                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        # Tăng sáng & tương phản cho các khung hình tối/CGI
                        enhancer_b = ImageEnhance.Brightness(img)
                        img_bright = enhancer_b.enhance(1.3)
                        enhancer_c = ImageEnhance.Contrast(img_bright)
                        img_final = enhancer_c.enhance(1.1)

                        images.append(img_final)
                        valid_paths.append(p)
                    except Exception as e:
                        print(f"    ⚠️ Lỗi đọc ảnh {p}: {e}")

                if not images:
                    continue

                prompts = [
                    "Question: Provide a highly detailed visual description of this scene. "
                    "You must explicitly describe the main physical actions, specific objects, "
                    "exact colors of clothing, the background environment, and any visible text or logos. "
                    "Answer: The scene shows"
                ] * len(images)

                dtype = torch.float16 if self.device == "cuda" else torch.float32
                inputs = self.processor(images=images, text=prompts, return_tensors="pt").to(self.device, dtype)

                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=65,
                    min_new_tokens=20,
                    repetition_penalty=1.15
                )
                generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

                for path, text in zip(valid_paths, generated_texts):
                    file_name = os.path.basename(path)
                    clean_text = clean_and_validate_caption(text)

                    if clean_text:
                        f.write(f"{file_name} | {clean_text}\n")
                    else:
                        f.write(f"{file_name} | The scene shows a dark or fast-moving physical action.\n")

        print(f"  ✅ Đã lưu captions thành công vào {output_file}")
        return output_file

    def generate_all_captions(
        self,
        data_dir: str,
        batch_size: int = 16,
        overwrite: bool = False
    ):
        """Quét toàn bộ các thư mục phim và sinh caption bằng VLM."""
        if not os.path.exists(data_dir):
            print(f"[-] LỖI: Thư mục dữ liệu {data_dir} không tồn tại.")
            return

        movie_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        print(f"\n🚀 BẮT ĐẦU QUY TRÌNH VLM CAPTIONING CHO {len(movie_folders)} PHIM...")

        for movie_name in movie_folders:
            movie_path = os.path.join(data_dir, movie_name)
            self.process_movie(movie_path, movie_name, batch_size=batch_size, overwrite=overwrite)

        print("\n[+] HOÀN THÀNH QUY TRÌNH TRÍCH XUẤT CAPTION CHO TOÀN BỘ BỘ PHIM!\n")
