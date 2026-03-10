import streamlit as st
import pandas as pd
import os
import sys

# Add src to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.recommender import RecommenderSystem

# ----------------- CONFIGURATION -----------------
st.set_page_config(
    page_title="Amazon Recommender AI", 
    page_icon="🛒", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .product-card {
        border-radius: 10px;
        padding: 15px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        height: 100%;
    }
    .stImage > img {
        border-radius: 8px;
        object-fit: cover;
    }
    .price-tag {
        color: #B12704;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Datasets paths (matching the preprocessing scripts)
REVIEW_PATH = r"F:\amazon_data\data\csv\Home_and_Kitchen_review.csv"
META_PATH = r"F:\amazon_data\data\csv\meta_Home_and_Kitchen.csv"
MODEL_DIR = r"F:\amazon_data\models"

# ----------------- APP LOGIC -----------------
st.title("🛒 Amazon Recommendation AI")
st.markdown("Hệ thống gợi ý sản phẩm tự động sử dụng **Content-based Filtering** và **TF-IDF Vectorizer** dựa trên dữ liệu Amazon Reviews 2023.")

@st.cache_resource(show_spinner=False)
def load_recommender():
    recommender = RecommenderSystem(REVIEW_PATH, META_PATH)
    # Load offline model directly instead of raw csv
    recommender.load_model(MODEL_DIR)
    return recommender

# UI: Check if Data Exists
expected_files = ["reviews.pkl", "meta.pkl", "popular_items.pkl", "tfidf_matrix.npz"]
missing_files = [f for f in expected_files if not os.path.exists(os.path.join(MODEL_DIR, f))]

if missing_files:
    st.error("⚠️ **Chưa tìm thấy tập dữ liệu Mô Hình Tối Ưu (Offline Model).**")
    st.info(f"Vui lòng chạy lệnh: `python src/train.py` trước khi thiết lập web app để tạo ra các file mô hình siêu tốc trong `{MODEL_DIR}`.")
    st.stop()

# Load Data
with st.spinner("Đang tải dữ liệu và khởi tạo mô hình Hệ Khuyến Nghị (AI)... Vui lòng đợi trong giây lát."):
    try:
        rec_sys = load_recommender()
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        st.stop()

# ----------------- MAIN UI -----------------
# Get active users (who have rated at least 3 items in our sample)
active_users = rec_sys.reviews.groupby('user_id').size()
active_users = active_users[active_users >= 3].index.tolist()

if not active_users:
    st.warning("Trong mẫu dữ liệu hiện tại, không có khách hàng nào có lịch sử ≥ 3 đánh giá. Vui lòng tăng giới hạn tải dữ liệu trong file app.py.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Giả lập Khách hàng")
    selected_user = st.selectbox(
        "Chọn một Khách hàng (User ID):",
        options=active_users[:50],  # Show 50 sample users
        help="Chọn một ID ngẫu nhiên để xem lịch sử mua sắm và kết quả gợi ý."
    )
    
    st.divider()
    st.markdown("**Thông tin Dataset:**")
    st.markdown(f"- **Reviews:** {len(rec_sys.reviews):,} dòng")
    st.markdown(f"- **Sản phẩm (Meta):** {len(rec_sys.meta):,} dòng")
    
    st.divider()
    st.caption("Developed with Streamlit & Scikit-Learn")

# Display User History
st.subheader(f"📖 Lịch sử Đánh giá của Khách hàng: `{selected_user}`")
history = rec_sys.get_user_history(selected_user)

if history.empty:
    st.info("Khách hàng này chưa có dữ liệu mua sắm chi tiết có hình ảnh.")
else:
    # Create columns for historical items
    tabs = st.tabs(["Hiển thị Lịch sử (Dạng Grid)", "Dữ liệu thô"])
    
    with tabs[0]:
        cols = st.columns(min(len(history), 5))
        for i, (_, row) in enumerate(history.head(5).iterrows()):
            with cols[i]:
                st.markdown('<div class="product-card">', unsafe_allow_html=True)
                st.image(row['image_url'], use_container_width=True)
                st.markdown(f"**{row['title'][:45]}...**")
                st.markdown(f"Đánh giá: {'⭐' * int(row['rating'])}")
                st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        st.dataframe(history[['parent_asin', 'title', 'rating', 'timestamp']])

st.divider()

# Display AI Recommendations
st.subheader("🤖 AI Khuyên Dùng (Dành riêng cho bạn)")

with st.spinner("Đang tính toán ma trận tương đồng (Cosine Similarity)..."):
    recommendations = rec_sys.recommend_content_based(selected_user, n=5)

if not recommendations.empty:
    rec_cols = st.columns(len(recommendations))
    for i, (_, row) in enumerate(recommendations.iterrows()):
        with rec_cols[i]:
            st.markdown('<div class="product-card" style="border-top: 4px solid #f5b041;">', unsafe_allow_html=True)
            st.image(row['image_url'], use_container_width=True)
            st.markdown(f"**{row['title'][:50]}...**")
            
            # Display price if available
            price = row.get('price')
            if pd.notna(price) and price != "":
                st.markdown(f'<p class="price-tag">${price}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="price-tag">Giá: Đang cập nhật</p>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Hệ thống chưa tìm được sản phẩm phù hợp. Hãy trải nghiệm Gợi ý Phổ biến!")
    # Fallback to Popular
    pop_recs = rec_sys.recommend_popular(n=5)
    if not pop_recs.empty:
        pop_cols = st.columns(5)
        for i, (_, row) in enumerate(pop_recs.iterrows()):
             with pop_cols[i]:
                st.image(row['image_url'], use_container_width=True)
                st.caption(f"**{row['title'][:45]}**")
