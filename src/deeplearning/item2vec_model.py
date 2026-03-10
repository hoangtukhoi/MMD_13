import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class Item2VecRecommender:
    def __init__(self, model_path=None):
        """
        Khởi tạo mô hình Item2Vec.
        Nếu truyền `model_path`, sẽ nạp mô hình đã train.
        """
        self.model = None
        self.item_metadata = {}
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def train(self, transactions, vector_size=100, window=5, min_count=2, epochs=10):
        """
        Huấn luyện mô hình Item2Vec trên tập các giỏ hàng (transactions).
        Mỗi `transaction` là một danh sách các Item ID.
        """
        logging.info(f"Bắt đầu huấn luyện Item2Vec với {len(transactions)} giỏ hàng...")
        
        # Word2Vec coi mỗi item (product ID) như một từ, mỗi giỏ hàng như một câu
        # Sử dụng thuật toán Skip-gram (sg=1) thường mang lại performance tốt hơn cho recommendation
        self.model = Word2Vec(
            sentences=transactions,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=1, # Skip-gram
            workers=4,
            epochs=epochs
        )
        logging.info("Huấn luyện hoàn tất!")
        
    def build_metadata_index(self, meta_df):
        """
        Xây dựng dict tra cứu nhanh metadata của item (Tên, Ảnh, Giá...)
        để phục vụ trả về thông tin chi tiết qua API.
        """
        logging.info("Đang xây dựng Metadata Index...")
        for _, row in meta_df.iterrows():
            item_id = str(row['parent_asin'])
            
            # Xử lý hình ảnh phụ trợ (fallback)
            import ast
            image_url = "https://via.placeholder.com/300"
            try:
                img_str = row.get('images', '')
                if isinstance(img_str, str) and img_str != "":
                    img_list = ast.literal_eval(img_str)
                    if isinstance(img_list, list) and len(img_list) > 0:
                        img_dict = img_list[0]
                        image_url = img_dict.get('hi_res') or img_dict.get('large') or image_url
            except:
                pass
                
            self.item_metadata[item_id] = {
                "id": item_id,
                "title": str(row.get('title', "Unknown")).strip()[:100] + "...",
                "price": row.get('price', "$19.99"),
                "image": image_url,
                "rating": row.get('average_rating', 0.0)
            }
            
    def save_model(self, model_dir):
        """Lưu trữ mô hình và metadata index"""
        os.makedirs(model_dir, exist_ok=True)
        # Lưu core model
        if self.model:
            self.model.save(os.path.join(model_dir, "item2vec.model"))
            
        # Lưu metadata lookup table
        with open(os.path.join(model_dir, "item_metadata.json"), "w", encoding='utf-8') as f:
            json.dump(self.item_metadata, f, ensure_ascii=False)
            
        logging.info(f"Đã lưu mô hình tại {model_dir}")
        
    def load_model(self, model_dir):
        """Nạp mô hình và metadata từ thư mục"""
        model_file = os.path.join(model_dir, "item2vec.model")
        metadata_file = os.path.join(model_dir, "item_metadata.json")
        
        if os.path.exists(model_file):
            self.model = Word2Vec.load(model_file)
            logging.info("Đã nạp Item2Vec model.")
            
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding='utf-8') as f:
                self.item_metadata = json.load(f)
            logging.info(f"Đã nạp metadata cho {len(self.item_metadata)} sản phẩm.")
            
    def recommend(self, current_item_id, top_n=5):
        """
        Dự đoán sản phẩm liên quan dựa trên khoảng cách vector
        trong không gian Embedding của Item2Vec.
        """
        if not self.model:
            return []
            
        # Kiểm tra item có nằm trong tập từ vựng của model hay không
        if current_item_id not in self.model.wv.key_to_index:
            logging.warning(f"Item {current_item_id} không tồn tại trong từ vựng OOV (Out Of Vocabulary).")
            return []
            
        # Lấy top N items tương đồng nhất (Cosine Similarity trên các Vector)
        similar_items = self.model.wv.most_similar(current_item_id, topn=top_n)
        
        # Enrich thêm metadata (thêm title, image...)
        recommendations = []
        for item_id, score in similar_items:
            meta = self.item_metadata.get(item_id, {"id": item_id, "title": f"Sản phẩm {item_id}", "image": "https://via.placeholder.com/150"})
            
            # Trả về cùng với confidence score (độ tương đồng)
            rec = meta.copy()
            rec['similarity_score'] = round(float(score), 4)
            recommendations.append(rec)
            
        return recommendations

if __name__ == "__main__":
    # Test script offline
    print("Mô-đun Item2Vec đã sẵn sàng tích hợp với Backend!")
