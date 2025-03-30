import pandas as pd
import numpy as np
import faiss
import streamlit as st
from rapidfuzz import process, fuzz

@st.cache_data
def load():
    file_parts = ["tracks_part1.csv", "tracks_part2.csv", "tracks_part3.csv", "tracks_part4.csv", "tracks_part5.csv"]
    df_list = [pd.read_csv(file) for file in file_parts]
    data = pd.concat(df_list, ignore_index=True)
    return data

data = load()

attributes = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
features = data[attributes].fillna(0)
features = (features - features.min()) / (features.max() - features.min())
featuresArray = np.array(features,dtype=np.float64)
searchIndex = faiss.IndexFlatL2(featuresArray.shape[1])
searchIndex.add(featuresArray)

def findSimilarSongs(title,suggestions=20):
    bestMatch, matchScore, _ = process.extractOne(title,data['name'],scorer = fuzz.ratio)
    if matchScore < 60:
        return [f" Song '{title}' not found."]
    
    position = data[data['name'] == bestMatch].index[0]
    vector = featuresArray[position].reshape(1,-1)

    _,foundIndices = searchIndex.search(vector,suggestions+1)
    recommendedSongs = [data.iloc[i]['name'] for i in foundIndices[0] if i!= position]

    return recommendedSongs


def musicRecommendationApp():
    st.markdown("""
        <style>
            body {
                background-color: #121212;
                color: white;
                font-family: Arial, sans-serif;
            }
            .title {
                text-align: center;
                font-size: 36px;
                font-weight: bold;
                color: #1DB954;
                margin-bottom: 20px;
            }
            .input-box {
                border: 2px solid #1DB954;
                padding: 10px;
                width: 50%;
                margin: auto;
                border-radius: 10px;
                text-align: center;
                font-size: 18px;
            }
            .song-list {
                display: flex;
                align-items: center;
                background: #1DB954;
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                color: white;
                font-size: 18px;
                transition: 0.3s;
            }
            .song-list:hover {
                background: #17a34a;
                transform: scale(1.02);
            }
            .song-icon {
                margin-right: 10px;
                font-size: 22px;
            }
            .button {
                background-color: #1DB954;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 18px;
                text-align: center;
                width: 50%;
                margin: auto;
                display: block;
            }
            .button:hover {
                background-color: #17a34a;
            }
        </style>
        <div class='title'> Spotify Music Recommendation System </div>
    """, unsafe_allow_html=True)

    title = st.text_input("Enter a Song Name:", placeholder = "Type here...")

    if st.button("🎧 Get Similar Songs"):
        recommendations = findSimilarSongs(title)
        st.markdown("Recommended Songs ")
        for song in recommendations:
            st.markdown(f"<div class='song-list'><span class='song-icon'>🎵</span>{song}</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    musicRecommendationApp()    
