import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import clip
import torch
from PIL import Image
import os
import psycopg2
from dotenv import load_dotenv
import numpy as np
import json
import matplotlib.pyplot as plt
load_dotenv()

# Create a connection to the PostgreSQL database
def dbconnection():
    try:
        connection = psycopg2.connect(
            dbname=os.getenv("DATABASE"),
            user=os.getenv("USER"),
            password=os.getenv("PASSWORD"),
            host=os.getenv("HOST"),
            port=os.getenv("PORT")
        )
        return connection
    except Exception as e:
        print(f"Error: Unable to connect to the database. {str(e)}")
        return None

# Load CLIP model and preprocessing function
def load_clip_model(device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

# Create the images table in PostgreSQL
def create_images_table():
    try:
        conn = dbconnection()
        if conn is not None:
            cursor = conn.cursor()
            try:
                cursor.execute("DROP TABLE IF EXISTS Face_Analyze")
                conn.commit()
                print("Existing table dropped successfully.")
            except Exception as e:
                print("No existing table to drop.")
            cursor.execute("""CREATE TABLE Face_Analyze (id SERIAL PRIMARY KEY,image_path TEXT NOT NULL,original_filename TEXT NOT NULL,user_given_name TEXT NOT NULL,embedding TEXT NOT NULL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            conn.commit()
            cursor.close()
            conn.close()
            return "Face_Analyze table created successfully"
        else:
            return "Failed to connect to the database"
    except Exception as e:
        print("Error in creating table: ", str(e))
        return None

# Get CLIP embedding for a single image
def get_image_embedding(image_path: str, model, preprocess, device: str) -> np.ndarray:
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

# Display image to user and get name input
def display_image_and_get_name(image_path: str, original_filename: str) -> str:
    try:
        image = Image.open(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Original filename: {original_filename}")
        plt.tight_layout()
        plt.show(block=False)
        print(f"\nShowing image: {original_filename}")
        print("Please look at the displayed image.")
        while True:
            user_name = input("Enter a name for this image (or 'skip' to skip this image): ").strip()
            if user_name:
                if user_name.lower() == 'skip':
                    plt.close()
                    return None
                else:
                    plt.close()
                    return user_name
            else:
                print("Please enter a valid name or 'skip' to skip this image.")                
    except Exception as e:
        print(f"Error displaying image {image_path}: {str(e)}")
        plt.close()
        return None

# Process all images in folder with user input and store in database
def store_images_with_user_input(image_folder: str):
    try:
        model, preprocess, device = load_clip_model()
        print(f"Using device: {device}")        
        image_extensions = ('jpg', 'jpeg', 'png', 'bmp', 'tiff')
        image_files = [f for f in os.listdir(image_folder)
                       if f.lower().endswith(image_extensions)]
        if not image_files:
            print("No images found in the folder.")
            return        
        print(f"Found {len(image_files)} images to process.")
        print("Note: Each image will be displayed for you to provide a name.\n")        
        conn = dbconnection()
        if conn is None:
            print("Failed to connect to database.")
            return        
        cursor = conn.cursor()
        processed_count = 0
        skipped_count = 0
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(image_folder, image_file)
            user_given_name = display_image_and_get_name(image_path, image_file)
            if user_given_name is None:
                print(f"Skipped: {image_file}")
                skipped_count += 1
                continue
            embedding = get_image_embedding(image_path, model, preprocess, device)
            if embedding is not None:
                embedding_json = json.dumps(embedding.tolist())
                query = """INSERT INTO Face_Analyze (image_path, original_filename, user_given_name, embedding) VALUES (%s, %s, %s, %s)"""
                cursor.execute(query, (image_path, image_file, user_given_name, embedding_json))
                conn.commit()                
                print(f"Successfully stored: '{user_given_name}' (original: {image_file})")
                processed_count += 1
            else:
                print(f"Failed to generate embedding for: {image_file}")
                skipped_count += 1        
        cursor.close()
        conn.close()
        print("Processing Complete")
        print(f"Successfully processed: {processed_count} images")
        print(f"Skipped: {skipped_count} images")
        print("All images have been processed!")
    except Exception as e:
        print(f"Error in storing images: {str(e)}")

# Function to view stored images and their names
def view_stored_images():
    try:
        conn = dbconnection()
        if conn is None:
            print("Failed to connect to database.")
            return
        cursor = conn.cursor()
        cursor.execute("""SELECT id, original_filename, user_given_name, created_at FROM images ORDER BY created_at DESC""")
        results = cursor.fetchall()
        if results:
            print("\n=== Stored Images ===")
            print(f"{'ID':<5} {'Original Filename':<25} {'User Given Name':<25} {'Created At':<20}")
            print("-" * 80)
            for row in results:
                print(f"{row[0]:<5} {row[1]:<25} {row[2]:<25} {str(row[3]):<20}")
        else:
            print("No images found in the database.")        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error viewing stored images: {str(e)}")

if __name__ == "__main__":    
    create_images_table()    
    image_folder = "images"
    if not os.path.exists(image_folder):
        print(f"Error: Image folder '{image_folder}' does not exist.")
        print("Please create the folder and add some images to it.")
    else:
        store_images_with_user_input(image_folder)
