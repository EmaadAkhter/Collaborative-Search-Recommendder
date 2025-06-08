import numpy as np
import pandas as pd
import re
import warnings
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')


class AnimeRecommendationSystem:
    """
    A collaborative filtering-based anime recommendation system using KNN
    """

    def __init__(self, anime_path, rating_path):
        """Initialize the recommendation system with data paths"""
        self.anime_path = anime_path
        self.rating_path = rating_path
        self.anime_df = None
        self.rating_df = None
        self.data = None
        self.data_pivot = None
        self.model_knn = None
        self.user_matrix = None

    def load_data(self):
        """Load anime and rating datasets"""
        self.anime_df = pd.read_csv(self.anime_path)
        self.rating_df = pd.read_csv(self.rating_path)
        print(f"Anime dataset shape: {self.anime_df.shape}")
        print(f"Rating dataset shape: {self.rating_df.shape}")

    def preprocess_data(self, min_ratings_per_user=50):
        """
        Merge and preprocess the data

        Args:
            min_ratings_per_user (int): Minimum number of ratings per user to include
        """
        # Merge datasets
        fulldata = pd.merge(self.anime_df, self.rating_df, on="anime_id", suffixes=[None, "_user"])
        fulldata = fulldata.rename(columns={"rating_user": "user_rating"})

        # Handle missing ratings (-1 means not rated)
        self.data = fulldata.copy()
        self.data["user_rating"].replace(to_replace=-1, value=np.nan, inplace=True)
        self.data = self.data.dropna(axis=0)

        # Filter users with minimum number of ratings
        selected_users = self.data["user_id"].value_counts()
        valid_users = selected_users[selected_users >= min_ratings_per_user].index
        self.data = self.data[self.data["user_id"].isin(valid_users)]

        # Clean anime names
        self.data["name"] = self.data["name"].apply(self._clean_text)

        print(f"Final dataset shape: {self.data.shape}")
        print(f"Number of unique users: {self.data['user_id'].nunique()}")
        print(f"Number of unique anime: {self.data['name'].nunique()}")

    def _clean_text(self, text):
        """Clean anime names by removing HTML entities and special characters"""
        if pd.isna(text):
            return text

        text = re.sub(r'&quot;', '', text)
        text = re.sub(r'.hack//', '', text)
        text = re.sub(r'&#039;', '', text)
        text = re.sub(r'A&#039;s', '', text)
        text = re.sub(r'I&#039;', 'I\'', text)
        text = re.sub(r'&amp;', 'and', text)
        return text

    def create_pivot_table(self):
        """Create user-anime rating pivot table"""
        self.data_pivot = self.data.pivot_table(
            index="name",
            columns="user_id",
            values="user_rating"
        ).fillna(0)

        print(f"Pivot table shape: {self.data_pivot.shape}")

    def train_model(self):
        """Train KNN model for user-based collaborative filtering"""
        # Transpose to get users as rows, anime as columns
        user_pivot = self.data_pivot.T

        # Create sparse matrix for efficiency
        self.user_matrix = csr_matrix(user_pivot.values)

        # Train KNN model
        self.model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model_knn.fit(self.user_matrix)

        print("KNN model trained successfully!")

    def find_similar_users(self, user_id, n_neighbors=30):
        """
        Find users similar to the given user

        Args:
            user_id: Target user ID
            n_neighbors (int): Number of similar users to find

        Returns:
            pd.DataFrame: DataFrame with similar users and their distances
        """
        user_pivot = self.data_pivot.T

        if user_id not in user_pivot.index:
            raise ValueError(f"User ID {user_id} not found in the dataset")

        # Find user index
        user_index = user_pivot.index.get_loc(user_id)

        # Find similar users
        distances, indices = self.model_knn.kneighbors(
            self.user_matrix[user_index].reshape(1, -1),
            n_neighbors=n_neighbors
        )

        # Create results DataFrame
        similar_users = []
        for i in range(1, len(distances.flatten())):  # Skip first (self)
            similar_users.append({
                "Similar User ID": user_pivot.index[indices.flatten()[i]],
                "Distance": distances.flatten()[i]
            })

        return pd.DataFrame(similar_users)

    def get_user_ratings(self, user_id):
        """
        Get all ratings for a specific user

        Args:
            user_id: Target user ID

        Returns:
            pd.DataFrame: User's rated anime with ratings
        """
        if user_id not in self.data_pivot.columns:
            raise ValueError(f"User ID {user_id} not found")

        user_ratings = self.data_pivot[user_id]
        non_zero_ratings = user_ratings[user_ratings != 0]

        rated_anime_df = pd.DataFrame({
            "Anime": non_zero_ratings.index,
            f"User {user_id} Rating": non_zero_ratings.values
        }).sort_values(by=f"User {user_id} Rating", ascending=False)

        return rated_anime_df.reset_index(drop=True)

    def generate_recommendations(self, user_id, n_neighbors=30, top_n=10):
        """
        Generate anime recommendations for a user

        Args:
            user_id: Target user ID
            n_neighbors (int): Number of similar users to consider
            top_n (int): Number of recommendations to return

        Returns:
            pd.DataFrame: Recommended anime with frequency and average ratings
        """
        # Find similar users
        similar_users_df = self.find_similar_users(user_id, n_neighbors)
        similar_user_ids = similar_users_df["Similar User ID"].tolist()

        # Get base user's ratings
        user_base_ratings = self.data_pivot[user_id]

        # Aggregate recommendations from similar users
        anime_counts = defaultdict(int)
        anime_ratings = defaultdict(list)

        for similar_user in similar_user_ids:
            if similar_user == user_id:
                continue

            sim_ratings = self.data_pivot[similar_user]

            # Find anime rated by similar user but not by base user
            new_recs = (sim_ratings != 0) & (user_base_ratings == 0)

            for anime in self.data_pivot.index[new_recs]:
                rating = sim_ratings[anime]
                anime_counts[anime] += 1
                anime_ratings[anime].append(rating)

        # Create recommendations DataFrame
        if not anime_counts:
            return pd.DataFrame(columns=["Anime", "Frequency", "Average Rating"])

        recommendation_df = pd.DataFrame({
            "Anime": list(anime_counts.keys()),
            "Frequency": list(anime_counts.values()),
            "Average Rating": [np.mean(ratings) for ratings in anime_ratings.values()]
        })

        # Sort by frequency and then by average rating
        recommendation_df = recommendation_df.sort_values(
            by=["Frequency", "Average Rating"],
            ascending=False
        ).reset_index(drop=True)

        return recommendation_df.head(top_n)

    def get_anime_genres(self, anime_names):
        """
        Get genres for a list of anime names

        Args:
            anime_names (list): List of anime names

        Returns:
            pd.DataFrame: DataFrame with anime names and their genres
        """
        genres_list = []

        for anime_name in anime_names:
            genre_info = self._get_genre_by_name(anime_name)
            if not genre_info.empty:
                genres_list.append({
                    'name': anime_name,
                    'genre': genre_info.iloc[0]['genre']
                })
            else:
                genres_list.append({
                    'name': anime_name,
                    'genre': 'Unknown'
                })

        return pd.DataFrame(genres_list)

    def _get_genre_by_name(self, name_query):
        """Helper function to find anime genre by name"""
        if not isinstance(name_query, str):
            name_query = str(name_query)

        # Exact match first
        names_lower = self.anime_df['name'].fillna('').str.lower()
        exact_matches = self.anime_df[names_lower == name_query.lower()]
        if not exact_matches.empty:
            return exact_matches[['name', 'genre']]

        # Partial match if no exact match
        partial_matches = self.anime_df[
            self.anime_df['name'].fillna('').str.contains(name_query, case=False, na=False)
        ]
        return partial_matches[['name', 'genre']]

    def get_recommendations_with_genres(self, user_id, n_neighbors=30, top_n=10):
        """
        Generate recommendations with genre information

        Args:
            user_id: Target user ID
            n_neighbors (int): Number of similar users to consider
            top_n (int): Number of recommendations to return

        Returns:
            pd.DataFrame: Recommendations with genres
        """
        # Get basic recommendations
        recommendations = self.generate_recommendations(user_id, n_neighbors, top_n)

        if recommendations.empty:
            return recommendations

        # Add genre information
        genres_df = self.get_anime_genres(recommendations["Anime"].tolist())

        # Merge recommendations with genres
        final_recommendations = recommendations.merge(
            genres_df,
            left_on='Anime',
            right_on='name',
            how='left'
        ).drop(columns=['name'])

        return final_recommendations


def main():
    """Example usage of the AnimeRecommendationSystem"""

    # Initialize the system
    anime_path = "path/to/your/anime.csv"
    rating_path = "path/to/your/rating.csv"
    recommender = AnimeRecommendationSystem(anime_path, rating_path)

    # Load and preprocess data
    recommender.load_data()
    recommender.preprocess_data(min_ratings_per_user=50)
    recommender.create_pivot_table()

    # Train the model
    recommender.train_model()

    # Get a random user for demonstration
    available_users = recommender.data_pivot.columns.tolist()
    user_id = np.random.choice(available_users)

    print(f"\nGenerating recommendations for User ID: {user_id}")

    # Show user's current ratings
    print(f"\nUser {user_id}'s rated anime:")
    user_ratings = recommender.get_user_ratings(user_id)
    print(user_ratings.head(10))

    # Generate recommendations with genres
    print(f"\nRecommendations for User {user_id}:")
    recommendations = recommender.get_recommendations_with_genres(user_id, top_n=100)
    print(recommendations)

    # Find similar users
    print(f"\nUsers similar to User {user_id}:")
    similar_users = recommender.find_similar_users(user_id, n_neighbors=10)
    print(similar_users.head())


if __name__ == "__main__":
    main()
