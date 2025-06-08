from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
import uvicorn
import numpy as np
import pandas as pd
from recommender import AnimeRecommendationSystem

# Paths
anime_path = "path/to/your/anime.csv"
rating_path = "path/to/your/rating.csv"

app = FastAPI()
templates = Jinja2Templates(directory="template")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load recommender system
recommender = AnimeRecommendationSystem(anime_path, rating_path)
recommender.load_data()
recommender.preprocess_data(min_ratings_per_user=50)
recommender.create_pivot_table()
recommender.train_model()

# Serve HTML homepage
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/user/{user_id}/ratings")
def get_user_ratings(user_id: int):
    try:
        ratings_df = recommender.get_user_ratings(user_id)
        return ratings_df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/user/{user_id}/recommendations")
def get_recommendations(user_id: int, top_n: Optional[int] = 10, n_neighbors: Optional[int] = 30):
    try:
        recs_df = recommender.get_recommendations_with_genres(user_id, n_neighbors=n_neighbors, top_n=top_n)
        return recs_df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
