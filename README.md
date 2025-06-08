# 🎌 AnimeVerse - Collaborative Anime Recommendation System

A sophisticated anime recommendation system that uses collaborative filtering with K-Nearest Neighbors (KNN) to suggest personalized anime recommendations based on user preferences and similar user behaviors.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-00a393.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Collaborative Filtering**: Uses KNN algorithm to find similar users and generate recommendations
- **Real-time Web Interface**: Beautiful, responsive web UI with animated elements
- **User Analytics**: View user's anime library with ratings and statistics
- **Genre Information**: Displays anime genres alongside recommendations
- **RESTful API**: Clean API endpoints for integration with other applications
- **Scalable Architecture**: Efficient sparse matrix operations for handling large datasets

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Emaad_Ansari/Collaborative-Search-Recommendder.git
   cd Collaborative-Search-Recommendder
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn pandas numpy scikit-learn scipy jinja2 python-multipart
   ```

3. **Download the dataset**
   
   Download the Anime Recommendations Database from Kaggle:
   - [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
   
   Extract the files and update the paths in `main.py`:
   ```python
   anime_path = "path/to/your/anime.csv"
   rating_path = "path/to/your/rating.csv"
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8000` to access the web interface.

## 📊 Dataset

The system uses the Anime Recommendations Database which contains:

- **anime.csv**: Information about 12,294 anime titles
  - anime_id, name, genre, type, episodes, rating, members

- **rating.csv**: User ratings for anime
  - user_id, anime_id, rating (1-10 scale, -1 for not rated)

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI        │    │   ML Engine     │
│   (HTML/JS)     │◄──►│   Backend        │◄──►│   (KNN Model)   │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **AnimeRecommendationSystem** (`recommender.py`)
   - Data preprocessing and cleaning
   - KNN model training with cosine similarity
   - User-based collaborative filtering
   - Genre information integration

2. **FastAPI Backend** (`main.py`)
   - RESTful API endpoints
   - CORS configuration
   - Template rendering

3. **Web Interface** (`template/index.html`)
   - Responsive design with animated particles
   - Real-time recommendations
   - User analytics dashboard

## 🔧 API Endpoints

### Get User Ratings
```http
GET /api/user/{user_id}/ratings
```
Returns all anime rated by the specified user.

**Response:**
```json
[
  {
    "Anime": "Death Note",
    "User {user_id} Rating": 9
  }
]
```

### Get Recommendations
```http
GET /api/user/{user_id}/recommendations?top_n=10&n_neighbors=30
```
Returns personalized anime recommendations for the user.

**Parameters:**
- `top_n` (optional): Number of recommendations to return (default: 10)
- `n_neighbors` (optional): Number of similar users to consider (default: 30)

**Response:**
```json
[
  {
    "Anime": "Attack on Titan",
    "Frequency": 5,
    "Average Rating": 8.7,
    "genre": "Action, Drama, Fantasy"
  }
]
```

## 🧠 Algorithm Details

### Collaborative Filtering Process

1. **Data Preprocessing**
   - Filter users with minimum 50 ratings for quality
   - Clean anime names and handle missing values
   - Create user-anime rating matrix

2. **Model Training**
   - Convert to sparse matrix for efficiency
   - Train KNN model with cosine similarity
   - Optimize for large-scale recommendations

3. **Recommendation Generation**
   - Find K most similar users
   - Aggregate ratings from similar users
   - Rank by frequency and average rating
   - Filter out already-rated anime

### Performance Optimizations

- **Sparse Matrix Operations**: Efficient memory usage for large datasets
- **Cosine Similarity**: Robust similarity metric for user preferences
- **Caching**: Preprocessed data structures for fast recommendations

## 🎨 Web Interface Features

- **Animated Background**: Floating particles for visual appeal
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Loading**: Smooth loading animations and feedback
- **User Statistics**: Comprehensive analytics of user's anime library
- **Genre Tags**: Color-coded genre information
- **Rating Visualization**: Intuitive rating display with color coding

## 📈 Usage Examples

### Command Line Usage

```python
from recommender import AnimeRecommendationSystem

# Initialize the system
recommender = AnimeRecommendationSystem("anime.csv", "rating.csv")

# Load and prepare data
recommender.load_data()
recommender.preprocess_data(min_ratings_per_user=50)
recommender.create_pivot_table()
recommender.train_model()

# Get recommendations for user
user_id = 123
recommendations = recommender.get_recommendations_with_genres(user_id, top_n=10)
print(recommendations)
```

### Web Interface Usage

1. Enter a User ID (e.g., 3, 123, 456)
2. Click "Get Recommendations"
3. View your anime library and personalized recommendations
4. Explore genres and ratings

## 🛠️ Configuration

### Adjustable Parameters

- **min_ratings_per_user**: Minimum ratings required per user (default: 50)
- **n_neighbors**: Number of similar users to consider (default: 30)
- **top_n**: Number of recommendations to return (default: 10)

### Customization

You can modify the recommendation algorithm by:
- Changing similarity metrics in the KNN model
- Adjusting user filtering criteria
- Implementing hybrid recommendation approaches
- Adding content-based filtering features

## 📁 Project Structure

```
Collaborative-Search-Recommendder/
├── main.py                 # FastAPI application entry point
├── recommender.py          # Core recommendation system
├── template/
│   └── index.html         # Web interface
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── LICENSE               # MIT License
└── README.md             # This file
```

## 🧪 Testing

### Test with Sample Users

The system works with any valid user ID from the dataset. Try these example user IDs:
- User ID: 3
- User ID: 123
- User ID: 456
- User ID: 1000

### Expected Output

For each user, you'll see:
- User's anime library with ratings
- Personalized recommendations based on similar users
- Genre information for each anime
- Rating statistics and analytics

## 🔧 Troubleshooting

### Common Issues

1. **Dataset Path Error**
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   **Solution**: Update the file paths in `main.py` to point to your downloaded CSV files.

2. **User Not Found**
   ```
   {"error": "User ID not found in the dataset"}
   ```
   **Solution**: Use a valid user ID that exists in the rating dataset.

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Increase the `min_ratings_per_user` parameter to reduce dataset size.

### Performance Tips

- For faster loading, increase `min_ratings_per_user` to 100+
- Reduce `n_neighbors` for faster recommendations
- Use SSD storage for better I/O performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include error handling and logging
- Write unit tests for new features
- Update documentation for API changes

## 📚 Technical Details

### Dependencies

- **FastAPI**: Modern web framework for building APIs
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **scipy**: Scientific computing utilities
- **jinja2**: Template engine for HTML rendering

### System Requirements

- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 500MB for dataset and dependencies
- **CPU**: Multi-core processor recommended for faster processing

## 🔬 Future Enhancements

- [ ] Content-based filtering integration
- [ ] Deep learning recommendation models
- [ ] Real-time collaborative filtering
- [ ] User authentication and profiles
- [ ] Anime image integration
- [ ] Advanced analytics dashboard
- [ ] Mobile app development
- [ ] Social features (friends, sharing)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) by Cooper Union
- FastAPI framework for the robust backend
- scikit-learn for machine learning algorithms
- The anime community for providing ratings data

## 📞 Contact

**Emaad Ansari** - [GitHub Profile](https://github.com/Emaad_Ansari)

Project Link: [https://github.com/Emaad_Ansari/Collaborative-Search-Recommendder](https://github.com/Emaad_Ansari/Collaborative-Search-Recommendder)

---

⭐ If you found this project helpful, please give it a star on GitHub!
