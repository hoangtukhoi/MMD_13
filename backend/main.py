from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Amazon Recommendation Engine API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục chứa dữ liệu tĩnh đã cache
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Backend FastAPI đang chạy!"}

@app.get("/api/seed")
def get_seed_products():
    """
    API lấy danh sách 20 sản phẩm hạt giống (Global Bestsellers / Top items)
    đã được tiền xử lý bằng Apache Spark (Big Data).
    Khắc phục bài toán Cold Start cho người dùng mới.
    """
    seed_file_path = os.path.join(DATA_DIR, "seed_products.json")
    
    if not os.path.exists(seed_file_path):
        # Trả về lỗi nếu file chưa được sinh ra từ pipeline Spark
        raise HTTPException(
            status_code=404, 
            detail="Chưa tìm thấy dữ liệu hạt giống. Vui lòng chạy job PySpark trước (src/bigdata/extract_seed.py)"
        )
        
    try:
        with open(seed_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"status": "success", "source": "PySpark Offline Batch", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc dữ liệu: {str(e)}")


# Data models for the recommendation endpoint
class RecommendRequest(BaseModel):
    basket_item_ids: List[str]

@app.post("/api/recommend")
def get_recommendations(request: RecommendRequest):
    """
    API nhận danh sách các sản phẩm đang có trong giỏ hàng (basket_item_ids) 
    và trả về danh sách gợi ý mua kèm dựa trên Deep Learning (Real-time).
    """
    basket_items = request.basket_item_ids
    
    if not basket_items:
         raise HTTPException(status_code=400, detail="Giỏ hàng trống. Không thể gợi ý.")
         
    # Tích hợp mô hình Deep Learning (Item2Vec) real-time
    item2vec_dir = os.path.join(SCRIPT_DIR, "models", "item2vec")
    
    # 1. Khởi tạo và nạp Model nếu tồn tại
    if not os.path.exists(item2vec_dir):
        # Fallback về Mock Data nếu chưa train model
        return {
            "status": "success", 
            "message": "Model chưa được huấn luyện. Vui lòng chạy src/deeplearning/train_dl.py",
            "input_items": basket_items,
            "recommendations": []
        }
        
    # Import Item2VecRecommender linh hoạt
    import sys
    sys.path.append(os.path.join(SCRIPT_DIR, ".."))
    from src.deeplearning.item2vec_model import Item2VecRecommender
    
    # 2. Xử lý logic Real-time Recommendation
    try:
        # Tạm thời nạp lại model trong mỗi request (Trong thực tế nên cache vào RAM lúc app startup)
        recommender = Item2VecRecommender(model_path=item2vec_dir)
        
        # Chỉ lấy sản phẩm cuối cùng khách vừa bỏ vào giỏ làm Anchor item
        anchor_item = basket_items[-1]
        
        # Gọi hàm recommend từ thuật toán Word2Vec (Skip-gram Cosine Similarity)
        recs = recommender.recommend(current_item_id=anchor_item, top_n=3)
        
        # Format lại kết quả cho Frontend
        recommendations = []
        for i, rec in enumerate(recs):
            recommendations.append({
                "id": rec.get("id", f"ds_{i}"),
                "title": rec.get("title"),
                "price": f"${rec.get('price', '19.99')}",
                "image": rec.get("image"),
                "confidence": round(rec.get("similarity_score", 0.8) * 100, 1)
            })
            
        return {
            "status": "success", 
            "message": f"Dự đoán dựa trên sản phẩm cuối ({anchor_item})",
            "input_items": basket_items,
            "recommendations": recommendations
        }
    except Exception as e:
        print(f"Error during DL prediction: {e}")
        raise HTTPException(status_code=500, detail="Lỗi trong quá trình suy luận mô hình DL.")
    
if __name__ == "__main__":
    import uvicorn
    # Chạy server FastAPI tại port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
