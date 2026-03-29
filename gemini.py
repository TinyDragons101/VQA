import os
import json
import time
import concurrent.futures
from typing import List, Dict, Any
from google import genai
from google.genai import types
from google.oauth2 import credentials
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm
from threading import Lock
import threading

api_call_count = 0
api_call_lock = Lock()

load_dotenv()

class GeminiVLCaptionModel:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
    ):
        """
        Khởi tạo Gemini API Client sử dụng SDK google-genai mới.
        """
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        access_token = os.getenv("GCP_ACCESS_TOKEN")

        if not all([project_id, location, access_token]):
            raise ValueError("❌ Thiếu cấu hình GCP trong file .env (Project, Location hoặc Token)")
        
        creds = credentials.Credentials(access_token)
        
        # Khởi tạo Client mới
        self.client = genai.Client(
            vertexai=True, 
            project=project_id, 
            location=location,
            credentials=creds
        )
        self.model_name = model_name
        
        print(f"✅ Gemini SDK (google-genai) loaded: {model_name}")

    def generate_questions(
        self,
        image_path: str,
        prompt: str = "Dựa trên hình ảnh, hãy tạo 7 câu hỏi về các chi tiết nghệ thuật và văn hóa."
    ) -> List[str]:
        """
        Tạo 5 câu hỏi VQA dưới dạng JSON Array sử dụng SDK mới.
        """
        try:
            # Mở ảnh bằng PIL
            image = load_and_resize_image(image_path, 512)
            
            # Cấu hình hệ thống và format đầu ra
            config = types.GenerateContentConfig(
                system_instruction="Bạn là chuyên gia nghiên cứu văn hóa Việt Nam. Hãy trả về kết quả duy nhất dưới dạng một JSON Array chứa các chuỗi văn bản (string). Không giải thích gì thêm.",
                temperature=0.4,
                response_mime_type="application/json",
            )

            # Gọi API (Thứ tự trong SDK mới là nội dung trước, config sau)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=config
            )

            increase_api_count()

            # Xử lý text trả về
            if not response.text:
                return []

            # Parse JSON an toàn
            data = json.loads(response.text.strip())
            
            # Xử lý trường hợp model trả về object lồng list
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list):
                        return val[:8]
            
            return data[:8] if isinstance(data, list) else []

        except Exception as e:
            # Xử lý lỗi Rate Limit (429) hoặc lỗi kết nối
            if "429" in str(e):
                print(f"[!] Rate limit hit for {image_path}, waiting...")
                time.sleep(5)
            else:
                print(f"[-] Error processing {image_path}: {e}")
            return []

    def generate_questions_batch(self, image_paths, prompts, max_workers=2):
        """
        Xử lý song song bằng ThreadPoolExecutor.
        """
        results = [None] * len(image_paths)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_questions, path, p): i 
                for i, (path, p) in enumerate(zip(image_paths, prompts))
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results
    
    def generate_answers(self, image_path: str, prompt: str) -> List[List[str]]:
        """
        Trả về format: [["Câu hỏi 1", "Câu trả lời 1"], ["Câu hỏi 2", "Câu trả lời 2"]]
        """
        try:
            image = load_and_resize_image(image_path, 512)
            
            config = types.GenerateContentConfig(
                system_instruction=(
                    "Bạn là chuyên gia phân tích báo chí. "
                    "Hãy lọc các câu hỏi liên quan, trả lời dựa trên bài báo và hình ảnh. "
                    "Sắp xếp theo độ quan trọng giảm dần. "
                    "TRẢ VỀ DUY NHẤT một JSON Array theo định dạng: [[\"câu hỏi\", \"câu trả lời\"], [...]]"
                ),
                temperature=0.2, # Thấp để đảm bảo tính xác thực
                response_mime_type="application/json",
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=config
            )

            increase_api_count()

            if not response.text:
                return []

            # Parse kết quả
            data = json.loads(response.text.strip())
            
            # Nếu model trả về object lồng (ví dụ {"qa_pairs": [[...]]})
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list): return val
            
            return data if isinstance(data, list) else []

        except Exception as e:
            if "429" in str(e):
                time.sleep(10)
            print(f"[-] Error answers {os.path.basename(image_path)}: {e}")
            return []

    def generate_answers_batch(self, tasks, max_workers=5):
        results = [None] * len(tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_answers, t['image_path'], t['prompt']): i 
                for i, t in enumerate(tasks)
            }
            # Sử dụng tqdm để theo dõi tiến độ gọi API
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(tasks), desc="Gemini API Answering"):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

def load_and_resize_image(path, max_size=512):
    image = Image.open(path).convert("RGB")
    
    # Resize giữ aspect ratio
    image.thumbnail((max_size, max_size))
    
    return image

def increase_api_count():
    global api_call_count
    
    with api_call_lock:
        api_call_count += 1
        
        if api_call_count % 10 == 0:
            print(f"[API CALL] Total: {api_call_count} | Thread: {threading.current_thread().name}")