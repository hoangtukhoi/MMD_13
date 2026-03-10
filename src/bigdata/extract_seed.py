import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, round, desc, first, max, explode
import json
import uuid

def process_big_data():
    print("Khởi tạo SparkSession...")
    # Tạo Spark Session với cấu hình bổ sung memory nếu cần
    spark = SparkSession.builder \
        .appName("Amazon_Seed_Extractor") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
        
    # Cấu hình log level để tránh spam terminal
    spark.sparkContext.setLogLevel("WARN")

    # Đường dẫn file dữ liệu (phù hợp với cấu trúc file csv đang có)
    # Lưu ý: PySpark có thể đọc file phân tán nên sử dụng file CSV lớn trực tiếp
    # Nếu đang dùng file JSONL, tải nó thay vì CSV để tránh lỗi parser.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    review_path = os.path.join(project_root, "data", "csv", "Home_and_Kitchen_review.csv")
    meta_path = os.path.join(project_root, "data", "csv", "meta_Home_and_Kitchen.csv")
    
    if not os.path.exists(review_path) or not os.path.exists(meta_path):
        print(f"[LỖI] Không tìm thấy dữ liệu tại: {review_path} hoặc {meta_path}")
        return

    print("Đang đọc dữ liệu Reviews...")
    # Đọc dữ liệu review (có header)
    reviews_df = spark.read.csv(review_path, header=True, inferSchema=True)
    
    print("Đang đọc dữ liệu Metadata...")
    # Đọc dữ liệu meta
    # spark.read.csv với multiLine=True để tránh lỗi nếu mô tả có ký tự xuống dòng
    meta_df = spark.read.csv(meta_path, header=True, inferSchema=True, multiLine=True, escape='"')

    print("Đang chạy MapReduce Job bằng Spark SQL để tính toán...")
    # Tính số lượng đánh giá và trung bình đánh giá cho mỗi sản phẩm
    # group by theo parent_asin
    item_stats = reviews_df.groupBy("parent_asin").agg(
        count("rating").alias("rating_count"),
        round(avg("rating"), 2).alias("rating_avg")
    )
    
    # Lọc ra những sản phẩm chất lượng cao (rating_count > 10 và rating_avg >= 4.0)
    # Tăng rating_count threshold tuỳ thuộc vào dữ liệu lớn cỡ nào
    high_quality_items = item_stats.filter((col("rating_count") > 50) & (col("rating_avg") >= 4.0))
    
    print("Đang Join dữ liệu với Metadata để lấy thông tin sản phẩm...")
    # Kết nối với bảng meta để lấy thêm danh mục (categories), title, images
    joined_df = high_quality_items.join(meta_df, on="parent_asin", how="inner")
    
    # Sắp xếp theo mức độ phổ biến (số lượng mua bù đắp bằng số lượng đánh giá)
    sorted_items = joined_df.orderBy(desc("rating_count"))
    
    # Lấy ra Top N (Ví dụ ở đây lấy top 200 để từ đó random ra 20 hạt giống đa dạng sau, hoặc lấy thẳng top 20)
    # Vì mục tiêu là đa dạng danh mục, nên ta có thể phân nhóm theo "store" hoặc "categories" và lấy top của từng danh mục,
    # nhưng tạm thời hãy lấy Top 20 Best Sellers toàn cục trước.
    top_items = sorted_items.limit(20)
    
    # Thu thập dữ liệu về Driver Node (Pandas)
    print("Đang tải dữ liệu kết quả về Pandas DataFrame...")
    results_pd = top_items.toPandas()
    
    # Xử lý hình ảnh từ định dạng chuỗi của python dictionaries
    import ast
    def extract_image(img_str):
        if not isinstance(img_str, str) or img_str == "":
            return "https://via.placeholder.com/300"
        try:
            img_list = ast.literal_eval(img_str)
            if isinstance(img_list, list) and len(img_list) > 0:
                img_dict = img_list[0]
                return img_dict.get('hi_res') or img_dict.get('large') or "https://via.placeholder.com/300"
        except:
            return "https://via.placeholder.com/300"

    print("Đang trích xuất hình ảnh...")
    if 'images' in results_pd.columns:
        results_pd['image_url'] = results_pd['images'].apply(extract_image)
    else:
        results_pd['image_url'] = "https://via.placeholder.com/300"
        
    results_pd['title'] = results_pd['title'].fillna("Unknown Product")
    
    # Định dạng dữ liệu xuất JSON
    seed_products = []
    for _, row in results_pd.iterrows():
        seed_products.append({
            "id": str(row['parent_asin']), # Định nghĩa id
            "title": str(row['title']).strip()[:100] + "..." if len(str(row['title'])) > 100 else str(row['title']).strip(),
            "price": row['price'] if 'price' in row and pd.notna(row['price']) else "$19.99",
            "image": str(row['image_url']),
            "rating": float(row['rating_avg']),
            "reviews": int(row['rating_count']),
            "category": "Home & Kitchen" # Hardcode nếu không có cột category rõ ràng
        })
        
    print(f"Trích xuất thành công {len(seed_products)} sản phẩm hạt giống!")
    
    # Lưu ra thư mục backend/data để ứng dụng FastAPI / React vọc
    output_dir = os.path.join(project_root, "backend", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "seed_products.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(seed_products, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Đã lưu file caching hạt giống Big Data tại: {output_file}")
    
    # Dừng Spark
    spark.stop()

if __name__ == "__main__":
    import pandas as pd
    process_big_data()
