import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os

class RecommenderSystem:
    def __init__(self, review_path, meta_path):
        self.review_path = review_path
        self.meta_path = meta_path
        self.reviews = None
        self.meta = None
        self.popular_items = []
        
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = None

    def extract_image(self, img_str):
        """Extract the first high-res image URL from the string representation of a list of dicts"""
        if pd.isna(img_str):
            return "https://via.placeholder.com/150"
        try:
            img_list = ast.literal_eval(img_str)
            if isinstance(img_list, list) and len(img_list) > 0:
                img_dict = img_list[0]
                if isinstance(img_dict, dict):
                    if 'hi_res' in img_dict and img_dict['hi_res']:
                        return img_dict['hi_res']
                    elif 'large' in img_dict and img_dict['large']:
                        return img_dict['large']
        except:
            pass
        return "https://via.placeholder.com/150"

    def load_data(self, n_rows_reviews=500000, n_rows_meta=200000):
        if not os.path.exists(self.review_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Data files not found. Please run the data extraction scripts.")
            
        # Load data
        self.reviews = pd.read_csv(self.review_path, nrows=n_rows_reviews)
        self.meta = pd.read_csv(self.meta_path, nrows=n_rows_meta)
        
        # Clean specific columns
        self.meta['title'] = self.meta['title'].fillna("Unknown Product")
        if 'images' in self.meta.columns:
            self.meta['image_url'] = self.meta['images'].apply(self.extract_image)
        else:
            self.meta['image_url'] = "https://via.placeholder.com/150"
            
        # To save memory and compute time, keep only items in meta that are also in our subset of reviews
        valid_items = self.reviews['parent_asin'].unique()
        self.meta = self.meta[self.meta['parent_asin'].isin(valid_items)].copy()
        
        # Calculate popular items
        item_stats = self.reviews.groupby('parent_asin').agg(
            rating_count=('rating', 'count'),
            rating_mean=('rating', 'mean')
        ).reset_index()
        
        # Filter items with at least 5 reviews and good ratings
        popular = item_stats[(item_stats['rating_count'] >= 5) & (item_stats['rating_mean'] >= 4.0)]
        popular = popular.sort_values(by='rating_count', ascending=False)
        self.popular_items = popular['parent_asin'].tolist()

        # Build Content-Based TF-IDF matrix
        self.meta = self.meta.reset_index(drop=True)
        self.tfidf_matrix = self.tfidf.fit_transform(self.meta['title'])

    def load_model(self, model_dir):
        """Load pre-trained model artifacts from disk to drastically speed up app boot time."""
        import pickle
        from scipy import sparse
        
        required_files = ["reviews.pkl", "meta.pkl", "popular_items.pkl", "tfidf_matrix.npz"]
        
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing model file: {file_path}. Please run train.py first.")
                
        # Unpickle dataframes
        with open(os.path.join(model_dir, "reviews.pkl"), "rb") as f:
            self.reviews = pickle.load(f)
            
        with open(os.path.join(model_dir, "meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)
            
        with open(os.path.join(model_dir, "popular_items.pkl"), "rb") as f:
            self.popular_items = pickle.load(f)
            
        # Unpack sparse array
        self.tfidf_matrix = sparse.load_npz(os.path.join(model_dir, "tfidf_matrix.npz"))

    def get_user_history(self, user_id):
        user_reviews = self.reviews[self.reviews['user_id'] == user_id]
        if user_reviews.empty:
            return pd.DataFrame()
        
        # Merge with meta to get titles and images
        history = pd.merge(user_reviews, self.meta, on='parent_asin', how='inner')
        return history.sort_values(by='rating', ascending=False)

    def recommend_popular(self, n=5):
        if not self.popular_items:
            return pd.DataFrame()
            
        top_asins = self.popular_items[:n]
        recs = self.meta[self.meta['parent_asin'].isin(top_asins)].head(n)
        return recs

    def recommend_content_based(self, user_id, n=5):
        history = self.get_user_history(user_id)
        
        # If no history or no good ratings, fall back to popular
        if history.empty:
            return self.recommend_popular(n)
            
        liked_items = history[history['rating'] >= 4.0]
        if liked_items.empty:
            return self.recommend_popular(n)
            
        # Create a user profile based on titles of items they liked
        liked_asins = liked_items['parent_asin'].tolist()
        liked_indices = self.meta.index[self.meta['parent_asin'].isin(liked_asins)].tolist()
        
        if not liked_indices:
             return self.recommend_popular(n)

        # Average the TF-IDF vectors of liked items
        user_profile = np.asarray(self.tfidf_matrix[liked_indices].mean(axis=0))
        
        # Calculate cosine similarity with all items
        cosine_sim = cosine_similarity(user_profile, self.tfidf_matrix)
        
        # Get top indices
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Filter out items already reviewed by user
        user_reviewed_asins = set(history['parent_asin'])
        
        recs_indices = []
        for i, score in sim_scores:
            asin = self.meta.iloc[i]['parent_asin']
            if asin not in user_reviewed_asins:
                recs_indices.append(i)
            if len(recs_indices) >= n:
                break
                
        return self.meta.iloc[recs_indices]
