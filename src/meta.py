import pandas as pd
import os

# 1. Duong dan vao file Meta (Doi ten 'Gift_Cards' thanh danh muc ban muon)
meta_file_path = r"F:\amazon_data\meta_Home_and_Kitchen.jsonl"
output_meta_csv = r"F:\amazon_data\meta_Home_and_Kitchen.csv"

# Giam chunk_size xuong vi file meta chua nhieu text dai, de bi tran RAM
chunk_size = 20000 

# Xoa file csv cu neu da ton tai
if os.path.exists(output_meta_csv):
    os.remove(output_meta_csv)

print(f"Bat dau doc file Meta: {meta_file_path}")

# Danh sach cac cot ban muon lay de hien thi len giao dien (UI)
# Ban co the them 'description' hoac 'average_rating' neu can
cols_to_keep = ['parent_asin', 'title', 'price', 'images']

# 2. Vong lap doc tung chunk
for i, chunk in enumerate(pd.read_json(meta_file_path, lines=True, chunksize=chunk_size)):
    print(f"-> Dang xu ly chunk thu {i + 1}...")
    
    # Meo nho: Chi chon nhung cot thuc su ton tai trong chunk nay 
    # (De phong truong hop co chunk bi thieu du lieu khong bi vang loi KeyError)
    existing_cols = [col for col in cols_to_keep if col in chunk.columns]
    chunk = chunk[existing_cols]
    
    write_header = True if i == 0 else False
    
    # Ghi noi tiep xuong file CSV
    chunk.to_csv(output_meta_csv, mode='a', header=write_header, index=False)

print(f"\nHOAN THANH! Du lieu san pham da luu tai: {output_meta_csv}")