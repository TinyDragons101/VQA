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
        model_name: str = "gemini-2.5-flash-lite",
    ):
        """
        Khởi tạo Gemini API Client sử dụng SDK google-genai mới.
        """
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.access_token = os.getenv("GCP_ACCESS_TOKEN")
        self.model_name = model_name

        if not all([self.project_id, self.access_token]):
            raise ValueError("❌ Thiếu cấu hình GCP trong file .env (Project, Location hoặc Token)")
        self.creds = credentials.Credentials(token=self.access_token)
                
        # Khởi tạo Client mới
        self.model_name = model_name
        
        self.locations = [
            "us-central1", "us-east1", "us-east4", "us-east5", "us-south1", "us-west1", "us-west4",
            "europe-central2", "europe-north1", "europe-southwest1", "europe-west1", "europe-west4", "europe-west8", "europe-west9"
        ]   
        
        self.clients = []
        for loc in self.locations:
            client = genai.Client(
                vertexai=True, 
                project=self.project_id, 
                location=loc,
                credentials=self.creds
            )
            self.clients.append({"client": client, "location": loc})
        
        # Biến điều khiển Round Robin
        self.current_idx = 0
        self.rr_lock = Lock()
        
        print(f"✅ Gemini SDK (google-genai) loaded: {model_name}")
        print(f"✅ Total Regions Active: {len(self.locations)}")
    
    def _get_next_client_info(self):
        """Lấy Client tiếp theo theo cơ chế Round Robin."""
        with self.rr_lock:
            client_info = self.clients[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.clients)
            return client_info

    def generate_questions(
        self,
        image_path: str,
        prompt: str = "Dựa trên hình ảnh, hãy tạo 7 câu hỏi về các chi tiết nghệ thuật và văn hóa."
    ) -> List[str]:
        """
        Tạo 5 câu hỏi VQA dưới dạng JSON Array sử dụng SDK mới.
        """
        info = self._get_next_client_info()
        client = info["client"]
        loc = info["location"]
        
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
            response = client.models.generate_content(
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
                print(f"[!] Rate limit hit for {image_path}, waiting... Error: {e}")
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
        info = self._get_next_client_info()
        client = info["client"]
        loc = info["location"]
        
        try:
            image = load_and_resize_image(image_path, 512)
            
            config = types.GenerateContentConfig(
                system_instruction=(
                    "Bạn là một trợ lý AI phân tích hình ảnh và bài báo chuyên nghiệp."
                    "Trả lời dựa trên bài báo và hình ảnh. "
                    "TRẢ VỀ DUY NHẤT một JSON Array theo định dạng: [[\"câu hỏi\", \"câu trả lời\"], [...]]"
                ),
                temperature=0.2, # Thấp để đảm bảo tính xác thực
                response_mime_type="application/json",
            )
            
            print(f"[*] Generating answers for {os.path.basename(image_path)} using {loc}...")
            print(f"[*] Prompt length: {len(prompt)} characters")

            response = client.models.generate_content(
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

def load_and_resize_image(path, max_size=384):
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