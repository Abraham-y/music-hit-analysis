"""
Spotify Audio Feature Collection
Collects audio features for songs using Spotify Web API
"""

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
from tqdm import tqdm
import rapidfuzz
from rapidfuzz import fuzz
import json
import os

class SpotifyAudioCollector:
    def __init__(self, client_id, client_secret):
        """Initialize Spotify client"""
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
        )
        self.cache_file = "audio_raw.csv"
        self.failures_file = "audio_failures.csv"
        
    def search_track(self, title, artist, max_retries=3):
        """Search for track on Spotify with fuzzy matching fallback"""
        
        # Try exact search first
        for attempt in range(max_retries):
            try:
                query = f"track:{title} artist:{artist}"
                results = self.sp.search(q=query, type='track', limit=10)
                
                if results['tracks']['items']:
                    # Find best match among results
                    best_match = None
                    best_score = 0
                    
                    for track in results['tracks']['items']:
                        # Get track info
                        track_title = track['name'].lower()
                        track_artists = [a['name'].lower() for a in track['artists']]
                        
                        # Calculate similarity scores
                        title_score = fuzz.ratio(title, track_title) / 100
                        artist_score = max(fuzz.ratio(artist, a) / 100 for a in track_artists)
                        
                        # Combined score (weighted more toward title)
                        combined_score = (title_score * 0.7 + artist_score * 0.3)
                        
                        if combined_score > best_score and combined_score > 0.85:
                            best_score = combined_score
                            best_match = track
                    
                    if best_match:
                        return best_match
                        
                # If no good match, try broader search
                if attempt == 0:
                    query = f"{title} {artist}"
                elif attempt == 1:
                    query = title
                else:
                    break
                    
                results = self.sp.search(q=query, type='track', limit=5)
                if results['tracks']['items']:
                    return results['tracks']['items'][0]
                    
            except Exception as e:
                print(f"Search error for {title} by {artist}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                break
        
        return None
    
    def get_audio_features(self, track_id):
        """Get audio features for a track"""
        try:
            features = self.sp.audio_features([track_id])
            if features and features[0]:
                return features[0]
        except Exception as e:
            print(f"Error getting audio features for {track_id}: {e}")
        return None
    
    def process_songs(self, songs_df, batch_size=50):
        """Process all songs and collect audio features"""
        
        # Load existing cache if exists
        if os.path.exists(self.cache_file):
            cache_df = pd.read_csv(self.cache_file)
            processed_titles = set(zip(cache_df['title'], cache_df['artist']))
            print(f"Loaded {len(cache_df)} cached results")
        else:
            cache_df = pd.DataFrame()
            processed_titles = set()
        
        # Filter songs not yet processed
        remaining_songs = songs_df[
            ~songs_df.set_index(['title', 'artist']).index.isin(processed_titles)
        ].copy()
        
        print(f"Processing {len(remaining_songs)} remaining songs...")
        
        results = []
        failures = []
        
        for idx, row in tqdm(remaining_songs.iterrows(), total=len(remaining_songs), desc="Collecting audio features"):
            title = row['title']
            artist = row['artist']
            
            # Search for track
            track = self.search_track(title, artist)
            
            if track:
                # Get audio features
                features = self.get_audio_features(track['id'])
                
                if features:
                    result = {
                        'title': title,
                        'artist': artist,
                        'year': row['year'],
                        'chart_position': row['chart_position'],
                        'weeks_on_chart': row['weeks_on_chart'],
                        'spotify_id': track['id'],
                        'spotify_title': track['name'],
                        'spotify_artists': ', '.join([a['name'] for a in track['artists']]),
                        'danceability': features['danceability'],
                        'energy': features['energy'],
                        'valence': features['valence'],
                        'tempo': features['tempo'],
                        'acousticness': features['acousticness'],
                        'instrumentalness': features['instrumentalness'],
                        'speechiness': features['speechiness'],
                        'loudness': features['loudness'],
                        'duration_ms': features['duration_ms'],
                        'match_score': 'found'
                    }
                    results.append(result)
                else:
                    failures.append({
                        'title': title,
                        'artist': artist,
                        'year': row['year'],
                        'error': 'audio_features_failed'
                    })
            else:
                failures.append({
                    'title': title,
                    'artist': artist,
                    'year': row['year'],
                    'error': 'track_not_found'
                })
            
            # Rate limiting
            if idx % 10 == 0:
                time.sleep(1)
        
        # Save results
        if results:
            new_results_df = pd.DataFrame(results)
            cache_df = pd.concat([cache_df, new_results_df], ignore_index=True)
            cache_df.to_csv(self.cache_file, index=False)
            print(f"Saved {len(results)} new audio features")
        
        if failures:
            failures_df = pd.DataFrame(failures)
            failures_df.to_csv(self.failures_file, index=False)
            print(f"Logged {len(failures)} failures")
        
        return cache_df, failures_df
    
    def clean_audio_data(self, df):
        """Clean and validate audio feature data"""
        
        # Add decade column
        df['decade'] = (df['year'] // 10) * 10
        df['decade'] = df['decade'].astype(str) + 's'
        
        # Check for outliers and bad data
        print(f"Before cleaning: {len(df)} songs")
        
        # Remove songs with invalid tempo or duration
        df = df[(df['tempo'] > 0) & (df['tempo'] < 300)]  # Reasonable tempo range
        df = df[(df['duration_ms'] > 30000) & (df['duration_ms'] < 600000)]  # 30s to 10min
        
        # Check for null values in key features
        key_features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 
                       'instrumentalness', 'speechiness', 'loudness']
        
        null_counts = df[key_features].isnull().sum()
        if null_counts.any():
            print(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
            df = df.dropna(subset=key_features)
        
        print(f"After cleaning: {len(df)} songs")
        
        # Save clean version
        df.to_csv('audio_clean.csv', index=False)
        print("Saved audio_clean.csv")
        
        return df

def main():
    """Main execution function"""
    
    # Load songs data
    if not os.path.exists('songs.csv'):
        print("Error: songs.csv not found. Run Phase_0_Billboard_Data.ipynb first!")
        return
    
    songs_df = pd.read_csv('songs.csv')
    print(f"Loaded {len(songs_df)} songs from songs.csv")
    
    # Get Spotify credentials (you'll need to set these)
    # Register at: https://developer.spotify.com/dashboard
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', 'your_client_id_here')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', 'your_client_secret_here')
    
    if CLIENT_ID == 'your_client_id_here' or CLIENT_SECRET == 'your_client_secret_here':
        print("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables")
        print("Or modify the CLIENT_ID and CLIENT_SECRET variables in this script")
        return
    
    # Initialize collector
    collector = SpotifyAudioCollector(CLIENT_ID, CLIENT_SECRET)
    
    # Process songs
    audio_df, failures_df = collector.process_songs(songs_df)
    
    # Clean data
    clean_df = collector.clean_audio_data(audio_df)
    
    print(f"\nFinal results:")
    print(f"Successfully processed: {len(clean_df)} songs")
    print(f"Failed: {len(failures_df)} songs")
    print(f"Success rate: {len(clean_df) / len(songs_df) * 100:.1f}%")

if __name__ == "__main__":
    main()
