import pandas as pd
import os

# 1. Duong dan file
review_file_path = r"F:\amazon_data\data\json\review_Home_and_Kitchen.jsonl"
output_csv_path = r"F:\amazon_data\data\csv\Home_and_Kitchen_review.csv"
chunk_size = 100000 

# Xoa file csv cu neu da ton tai de tranh bi loi ghi noi tiep vao du lieu cu
if os.path.exists(output_csv_path):
    os.remove(output_csv_path)

print(f"Bat dau doc file: {review_file_path}")

# 2. Vong lap doc tung chunk va ghi truc tiep xuong o cung
for i, chunk in enumerate(pd.read_json(review_file_path, lines=True, chunksize=chunk_size)):
    print(f"-> Dang xu ly chunk thu {i + 1}...")
    
    # Loc cac cot can thiet
    chunk = chunk[['user_id', 'parent_asin', 'rating', 'timestamp']]
    
    # Kiem tra xem co phai chunk dau tien khong de quyet dinh viec in tieu de (header)
    # Chi in header 1 lan duy nhat o chunk dau tien (i = 0)
    write_header = True if i == 0 else False
    
    # Ghi noi tiep vao file CSV
    chunk.to_csv(output_csv_path, mode='a', header=write_header, index=False)

print(f"\nHOAN THANH! Du lieu da duoc luu tai: {output_csv_path}")