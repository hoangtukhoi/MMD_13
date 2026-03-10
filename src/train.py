import os
import sys
import pickle
from scipy import sparse

# Add src to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.recommender import RecommenderSystem

def main():
    print("Starting Model Training (Offline Training)...")
    
    # Paths matching app.py settings
    REVIEW_PATH = r"F:\amazon_data\data\csv\Home_and_Kitchen_review.csv"
    META_PATH = r"F:\amazon_data\data\csv\meta_Home_and_Kitchen.csv"
    MODEL_DIR = r"F:\amazon_data\models"
    
    # Ensure model directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print(f"Using Data:\n- {REVIEW_PATH}\n- {META_PATH}")
    recommender = RecommenderSystem(REVIEW_PATH, META_PATH)
    
    print("Reading data and computing TF-IDF... This may take a few seconds.")
    try:
        recommender.load_data(n_rows_reviews=200000, n_rows_meta=100000)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run download.py and preprocess.py first!")
        return

    print("Computation finished. Serializing to disk...")
    
    # Save pandas dataframes
    with open(os.path.join(MODEL_DIR, "reviews.pkl"), "wb") as f:
        pickle.dump(recommender.reviews, f)
    with open(os.path.join(MODEL_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(recommender.meta, f)
        
    # Save popular items list
    with open(os.path.join(MODEL_DIR, "popular_items.pkl"), "wb") as f:
        pickle.dump(recommender.popular_items, f)
        
    # Save TF-IDF matrix using scipy.sparse (much more efficient than pickle for matrices)
    sparse.save_npz(os.path.join(MODEL_DIR, "tfidf_matrix.npz"), recommender.tfidf_matrix)

    print(f"\n✅ Completed! Model files saved to: {MODEL_DIR}")
    print("You can now open 'streamlit run app.py' for super fast boot time.")

if __name__ == "__main__":
    main()
