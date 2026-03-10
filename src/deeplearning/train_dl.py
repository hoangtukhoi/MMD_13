import os
import sys
import pandas as pd
import logging
import ast

# Tự động ghim thư mục gốc (f:\amazon_data) vào Python Path để nó hiểu module 'src'
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.deeplearning.item2vec_model import Item2VecRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def extract_transactions(reviews_df):
    """
    Chuyển đổi dữ liệu user-item interactions thành danh sách các 'giỏ hàng' (transactions).
    Tuyệt nhất là gom theo (reviewerID, date) nhưng dữ liệu ngày tháng review có thể thưa.
    Giản lược: gom tất cả các sản phẩm một user từng review thành một transaction (session).
    """
    logging.info("Đang trích xuất transactions từ dữ liệu reviews...")
    # Lọc lấy các user có >= 3 reviews để có tính chuỗi
    user_counts = reviews_df['reviewerID'].value_counts()
    valid_users = user_counts[user_counts >= 3].index
    
    filtered_df = reviews_df[reviews_df['reviewerID'].isin(valid_users)]
    
    # Sắp xếp theo unixReviewTime để tạo sequence đúng thứ tự thời gian mua
    filtered_df = filtered_df.sort_values(by=['reviewerID', 'unixReviewTime'])
    
    # Gom nhóm các parent_asin (sản phẩm) theo reviewerID
    transactions = filtered_df.groupby('reviewerID')['parent_asin'].apply(list).tolist()
    logging.info(f"Đã tạo {len(transactions)} sequences (transactions).")
    return transactions

def main():
    print("="*50)
    print("BẮT ĐẦU HUẤN LUYỆN DEEP LEARNING MODEL (ITEM2VEC)")
    print("="*50)
    
    # Cấu hình đường dẫn
    DATA_DIR = r"F:\amazon_data\data\csv"
    REVIEW_PATH = os.path.join(DATA_DIR, "Home_and_Kitchen_review.csv")
    META_PATH = os.path.join(DATA_DIR, "meta_Home_and_Kitchen.csv")
    MODEL_DIR = r"F:\amazon_data\backend\models\item2vec"
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Đọc dữ liệu
    logging.info("Đang nạp dữ liệu từ CSV...")
    try:
        # Load một phần dữ liệu để train nhanh trong môi trường demo
        # Actual columns in CSV: ['user_id', 'parent_asin', 'rating', 'timestamp']
        reviews_df = pd.read_csv(REVIEW_PATH, nrows=500000, usecols=['user_id', 'parent_asin', 'timestamp'])
        
        # Đổi tên cột cho khớp logic bên dưới
        reviews_df = reviews_df.rename(columns={'user_id': 'reviewerID', 'timestamp': 'unixReviewTime'})
        
        meta_df = pd.read_csv(META_PATH, nrows=200000)
    except FileNotFoundError:
        logging.error("Không tìm thấy file CSV. Vui lòng chạy pipeline tải dữ liệu (download.py & preprocess.py) trước.")
        return

    # 2. Xử lý dữ liệu thành chuỗi giao dịch (Sessions/Transactions)
    transactions = extract_transactions(reviews_df)
    
    # 3. Khởi tạo và Huấn luyện mô hình Item2Vec
    recommender = Item2VecRecommender()
    
    # Train model (có thể tinh chỉnh tham số)
    recommender.train(
        transactions=transactions, 
        vector_size=64, 
        window=5, 
        min_count=2, 
        epochs=15
    )
    
    # 4. Xây dựng Data Index (Metadata) cho API trả về đầy đủ ảnh, giá, tên
    recommender.build_metadata_index(meta_df)
    
    # 5. Lưu trữ Model
    recommender.save_model(MODEL_DIR)
    
    print("="*50)
    print(f"✅ Hoàn tất! Mô hình đã được lưu tại: {MODEL_DIR}")
    print("Bạn có thể tích hợp đường dẫn này vào backend/main.py để gợi ý Real-time.")
    print("="*50)

if __name__ == "__main__":
    main()
