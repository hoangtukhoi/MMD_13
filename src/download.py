import os
import requests

# 1. Cai dat thu muc luu tru tai o F
SAVE_DIR = r"F:\amazon_data"

# Tao thu muc neu chua ton tai
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. Danh sach 33 danh muc cua Amazon Reviews 2023
CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", 
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books", 
    "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", 
    "Collectible_Coins", "Digital_Music", "Electronics", "Fine_Art", 
    "Gift_Cards", "Grocery_and_Gourmet_Food", "Handmade_Products", 
    "Health_and_Household", "Home_and_Kitchen", "Industrial_and_Scientific", 
    "Kindle_Store", "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", 
    "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Software", 
    "Sports_and_Outdoors", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games"
]

def download_file(url, save_path):
    """Ham tai file voi che do streaming de khong ton RAM"""
    if os.path.exists(save_path):
        print(f"[Bo qua] File da ton tai: {os.path.basename(save_path)}")
        return
        
    print(f"Dang tai: {os.path.basename(save_path)}...")
    try:
        # stream=True giup tai file dung luong lon ma khong dua het vao RAM
        with requests.get(url, stream=True) as r:
            r.raise_for_status() # Bao loi neu URL khong ton tai
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"-> Da luu thanh cong: {save_path}")
    except Exception as e:
        print(f"-> LOI khi tai {url}: {e}")
        # Xoa file loi dang tai do de lan sau tai lai tu dau
        if os.path.exists(save_path):
            os.remove(save_path)

# 3. Vong lap tai du lieu
for cat in CATEGORIES:
    print(f"\n--- Dang xu ly danh muc: {cat} ---")
    
    # Da sua: Bo duoi .gz khoi URL Hugging Face
    review_url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{cat}.jsonl"
    meta_url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{cat}.jsonl"
    
    # Da sua: Bo duoi .gz o ten file luu ve may
    review_path = os.path.join(SAVE_DIR, f"review_{cat}.jsonl")
    meta_path = os.path.join(SAVE_DIR, f"meta_{cat}.jsonl")
    
    # Tien hanh tai
    download_file(review_url, review_path)
    download_file(meta_url, meta_path)

print("\nHOAN THANH TAI TOAN BO DU LIEU!")