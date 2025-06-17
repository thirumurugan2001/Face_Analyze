import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import clip
import torch
from PIL import Image
from dotenv import load_dotenv
import numpy as np
from typing import List, Tuple
import os
import json
import base64
import tempfile
load_dotenv()
from helper import dbconnection

# Load CLIP model and preprocessing function
def load_clip_model(device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# Convert Base64 string to temporary image file
def base64_to_temp_image(base64_string: str, file_extension: str = "jpg") -> str:
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]            
        image_data = base64.b64decode(base64_string)
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f'.{file_extension.lower()}',
            prefix='temp_image_'
        )
        temp_file.write(image_data)
        temp_file.close()
        try:
            with Image.open(temp_file.name) as img:
                img.verify()
            return temp_file.name
        except Exception as e:
            os.unlink(temp_file.name)
            raise Exception(f"Invalid image data: {str(e)}")
    except Exception as e:
        print(f"Error converting Base64 to image: {str(e)}")
        return None

# Clean up temporary file
def cleanup_temp_file(file_path: str):
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error cleaning up temporary file: {str(e)}")

# Encode text query using CLIP
def encode_text_query(query: str, model, device: str) -> np.ndarray:
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().flatten()

# Get image embedding from file path
def get_image_embedding_from_path(image_path: str, model, preprocess, device: str) -> np.ndarray:
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error getting embedding for {image_path}: {str(e)}")
        return None

# Calculate cosine similarity between two vectors
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Search for similar images using Base64 image
def search_similar_images_by_base64(base64_image: str, top_k: int = 5, file_extension: str = "jpg") -> List[Tuple[str, str, float]]:
    temp_image_path = None
    try:
        temp_image_path = base64_to_temp_image(base64_image, file_extension)
        if temp_image_path is None:
            return []        
        model, preprocess, device = load_clip_model()
        query_embedding = get_image_embedding_from_path(temp_image_path, model, preprocess, device)
        if query_embedding is None:
            return []
        conn = dbconnection()
        if conn is None:
            print("Failed to connect to database.")
            return []
        cursor = conn.cursor()        
        cursor.execute("SELECT id, image_path, user_given_name, embedding FROM face_analyze")
        results = cursor.fetchall()        
        if not results:
            print("No images found in database.")
            cursor.close()
            conn.close()
            return []                
        similarities = []
        for img_id, img_path, img_name, embedding_json in results:
            try:
                embedding_list = json.loads(embedding_json)
                img_embedding = np.array(embedding_list, dtype=np.float32)
                similarity = cosine_similarity(query_embedding, img_embedding)
                similarities.append((img_path, img_name, float(similarity)))
            except Exception as e:
                print(f"Error processing embedding for {img_name}: {str(e)}")
                continue        
        similarities.sort(key=lambda x: x[2], reverse=True)
        cursor.close()
        conn.close()
        return similarities[:top_k]
    except Exception as e:
        print(f"Error in Base64 image similarity search: {str(e)}")
        return []
    finally:
        if temp_image_path:
            cleanup_temp_file(temp_image_path)

# Search and return user_given_name for Base64 image
def search_and_return_base64_results(base64_image: str, top_k: int = 5, file_extension: str = "jpg"):
    results = search_similar_images_by_base64(base64_image, top_k, file_extension)
    if not results:
        print("No similar images found.")
        return None
    if results:
        best_match_name = results[0][1]
        return best_match_name
    return None