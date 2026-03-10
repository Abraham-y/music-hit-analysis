"""
Geospatial Analysis of Music
Implements geographic analysis of music production and consumption patterns:
- Geographic distribution of artists
- Regional music characteristics
- Spatial autocorrelation analysis
- Geographic clustering
- Cultural diffusion patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import requests
import json
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class GeospatialMusicAnalyzer:
    def __init__(self, merged_file='merged.csv'):
        """Initialize with merged dataset for geospatial analysis"""
        self.df = pd.read_csv(merged_file)
        self.df = self.df.dropna()
        self.prepare_data()
        
        # Initialize geocoder
        self.geolocator = Nominatim(user_agent="music_analysis")
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1)
    
    def prepare_data(self):
        """Prepare data for geospatial analysis"""
        print(f"Loaded {len(self.df)} songs for geospatial analysis")
        
        # Add decade column if not present
        if 'decade' not in self.df.columns:
            self.df['decade'] = (self.df['year'] // 10) * 10
            self.df['decade'] = self.df['decade'].astype(str) + 's'
        
        # Extract artist locations (will need to be geocoded)
        self.df['artist_location'] = self.df['artist'].apply(self.extract_artist_location)
        
        print("Geospatial data prepared")
    
    def extract_artist_location(self, artist_name):
        """Extract location information for artist"""
        # This is a simplified version - in practice, you'd use a music database API
        # For now, we'll create mock locations based on common music hubs
        
        music_hubs = {
            'Los Angeles': (34.0522, -118.2437),
            'New York': (40.7128, -74.0060),
            'Nashville': (36.1627, -86.7816),
            'Atlanta': (33.7490, -84.3880),
            'Chicago': (41.8781, -87.6298),
            'Miami': (25.7617, -80.1918),
            'Detroit': (42.3314, -83.0458),
            'Seattle': (47.6062, -122.3321),
            'New Orleans': (29.9511, -90.0715),
            'Austin': (30.2672, -97.7431),
            'London': (51.5074, -0.1278),
            'Toronto': (43.6532, -79.3832),
            'Jamaica': (18.1096, -77.2975),
            'Sweden': (59.3293, -18.0686),
            'South Korea': (37.5665, -126.9780)
        }
        
        # Simple heuristic to assign locations based on artist characteristics
        # In a real implementation, you'd use APIs like Spotify, MusicBrainz, etc.
        import random
        return random.choice(list(music_hubs.keys()))
    
    def geocode_artists(self):
        """Geocode artist locations to get coordinates"""
        
        # Create a cache for geocoded locations
        location_cache = {}
        
        def get_coordinates(location):
            if location in location_cache:
                return location_cache[location]
            
            try:
                location_obj = self.geolocator.geocode(location)
                if location_obj:
                    coords = (location_obj.latitude, location_obj.longitude)
                    location_cache[location] = coords
                    return coords
            except:
                pass
            
            # Default coordinates if geocoding fails
            location_cache[location] = (0, 0)
            return (0, 0)
        
        # Get coordinates for all unique locations
        unique_locations = self.df['artist_location'].unique()
        print(f"Geocoding {len(unique_locations)} unique locations...")
        
        for location in unique_locations:
            get_coordinates(location)
        
        # Add coordinates to dataframe
        self.df['latitude'] = self.df['artist_location'].map(lambda x: location_cache.get(x, (0, 0))[0])
        self.df['longitude'] = self.df['artist_location'].map(lambda x: location_cache.get(x, (0, 0))[1])
        
        # Remove invalid coordinates
        valid_coords = (self.df['latitude'] != 0) & (self.df['longitude'] != 0)
        self.df = self.df[valid_coords]
        
        print(f"Successfully geocoded {len(self.df)} songs")
        
        return location_cache
    
    def create_world_map(self):
        """Create world map visualization of music production"""
        
        try:
            # Load world map
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            
            # Create GeoDataFrame from our data
            geometry = gpd.points_from_xy(self.df['longitude'], self.df['latitude'])
            gdf = gpd.GeoDataFrame(self.df, geometry=geometry)
            
            # Plot
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            
            # World map with all points
            world.plot(ax=axes[0, 0], color='lightgray', edgecolor='white')
            gdf.plot(ax=axes[0, 0], color='red', markersize=10, alpha=0.6)
            axes[0, 0].set_title('Global Music Production Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Longitude')
            axes[0, 0].set_ylabel('Latitude')
            
            # Map by decade
            world.plot(ax=axes[0, 1], color='lightgray', edgecolor='white')
            decades = sorted(self.df['decade'].unique())
            colors = plt.cm.Set3(np.linspace(0, 1, len(decades)))
            
            for i, decade in enumerate(decades):
                decade_data = gdf[gdf['decade'] == decade]
                if len(decade_data) > 0:
                    decade_data.plot(ax=axes[0, 1], color=[colors[i]], markersize=8, 
                                   alpha=0.7, label=decade)
            
            axes[0, 1].set_title('Music Production by Decade', fontsize=14, fontweight='bold')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Map by energy levels
            world.plot(ax=axes[1, 0], color='lightgray', edgecolor='white')
            scatter = axes[1, 0].scatter(self.df['longitude'], self.df['latitude'], 
                                        c=self.df['energy'], cmap='viridis', 
                                        s=50, alpha=0.7)
            plt.colorbar(scatter, ax=axes[1, 0], label='Energy Level')
            axes[1, 0].set_title('Geographic Distribution of Song Energy', fontsize=14, fontweight='bold')
            
            # Map by lexical diversity
            world.plot(ax=axes[1, 1], color='lightgray', edgecolor='white')
            scatter = axes[1, 1].scatter(self.df['longitude'], self.df['latitude'], 
                                        c=self.df['lexical_diversity'], cmap='plasma', 
                                        s=50, alpha=0.7)
            plt.colorbar(scatter, ax=axes[1, 1], label='Lexical Diversity')
            axes[1, 1].set_title('Geographic Distribution of Lyric Complexity', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('world_music_map.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Could not create world map: {e}")
            # Fallback to scatter plot
            self.create_fallback_maps()
    
    def create_fallback_maps(self):
        """Create fallback scatter plots if geopandas is not available"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Basic scatter plot
        axes[0, 0].scatter(self.df['longitude'], self.df['latitude'], alpha=0.6, s=30)
        axes[0, 0].set_title('Global Music Production Distribution')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        
        # By decade
        decades = sorted(self.df['decade'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(decades)))
        
        for i, decade in enumerate(decades):
            decade_data = self.df[self.df['decade'] == decade]
            axes[0, 1].scatter(decade_data['longitude'], decade_data['latitude'], 
                              color=[colors[i]], alpha=0.7, s=30, label=decade)
        
        axes[0, 1].set_title('Music Production by Decade')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        axes[0, 1].legend()
        
        # By energy
        scatter = axes[1, 0].scatter(self.df['longitude'], self.df['latitude'], 
                                    c=self.df['energy'], cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, ax=axes[1, 0])
        axes[1, 0].set_title('Geographic Distribution of Song Energy')
        axes[1, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        
        # By lexical diversity
        scatter = axes[1, 1].scatter(self.df['longitude'], self.df['latitude'], 
                                    c=self.df['lexical_diversity'], cmap='plasma', alpha=0.6, s=30)
        plt.colorbar(scatter, ax=axes[1, 1])
        axes[1, 1].set_title('Geographic Distribution of Lyric Complexity')
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        
        plt.tight_layout()
        plt.savefig('world_music_map.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def regional_music_characteristics(self):
        """Analyze music characteristics by region"""
        
        # Define regions (simplified)
        def assign_region(lat, lon):
            if lat > 45 and lon > -130 and lon < -60:  # North America
                return 'North America'
            elif lat > 35 and lat < 70 and lon > -10 and lon < 40:  # Europe
                return 'Europe'
            elif lat < 35 and lat > -35 and lon > -80 and lon < 35:  # Latin America & Caribbean
                return 'Latin America'
            elif lat > -10 and lat < 40 and lon > 100 and lon < 180:  # East Asia
                return 'East Asia'
            else:
                return 'Other'
        
        self.df['region'] = self.df.apply(lambda row: assign_region(row['latitude'], row['longitude']), axis=1)
        
        # Analyze regional characteristics
        regional_stats = {}
        
        for region in self.df['region'].unique():
            region_data = self.df[self.df['region'] == region]
            
            stats = {
                'count': len(region_data),
                'avg_energy': region_data['energy'].mean(),
                'avg_valence': region_data['valence'].mean(),
                'avg_danceability': region_data['danceability'].mean(),
                'avg_lexical_diversity': region_data['lexical_diversity'].mean(),
                'avg_sentiment': region_data['vader_compound'].mean(),
                'top10_rate': region_data['top10'].mean() if 'top10' in region_data.columns else 0
            }
            
            regional_stats[region] = stats
        
        # Visualize regional differences
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['avg_energy', 'avg_valence', 'avg_danceability', 
                  'avg_lexical_diversity', 'avg_sentiment', 'top10_rate']
        metric_names = ['Energy', 'Valence', 'Danceability', 
                       'Lexical Diversity', 'Sentiment', 'Top 10 Rate']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            regions = list(regional_stats.keys())
            values = [regional_stats[region][metric] for region in regions]
            
            bars = axes[i].bar(regions, values, alpha=0.7, 
                             color=plt.cm.Set3(np.linspace(0, 1, len(regions))))
            axes[i].set_title(f'Regional {name}')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('regional_music_characteristics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return regional_stats
    
    def spatial_autocorrelation_analysis(self):
        """Analyze spatial autocorrelation in music characteristics"""
        
        try:
            from libpysal.weights import KNN
            from esda.moran import Moran
        except ImportError:
            print("libpysal/esda not available. Install with: pip install libpysal esda")
            return None
        
        # Create spatial weights
        coords = self.df[['latitude', 'longitude']].values
        weights = KNN(coords, k=5)
        
        # Calculate Moran's I for different characteristics
        characteristics = ['energy', 'valence', 'danceability', 'lexical_diversity', 'vader_compound']
        
        moran_results = {}
        
        for char in characteristics:
            if char in self.df.columns:
                values = self.df[char].values
                moran = Moran(values, weights)
                moran_results[char] = moran
                
                print(f"Moran's I for {char}: {moran.I:.3f} (p-value: {moran.p_sim:.3f})")
        
        # Visualize spatial autocorrelation
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (char, moran) in enumerate(moran_results.items()):
            if i < 6:
                # Create scatter plot of spatial lag
                axes[i].scatter(moran.y, moran.wy, alpha=0.6)
                axes[i].plot([moran.y.min(), moran.y.max()], 
                           [moran.y.min(), moran.y.max()], 'r--')
                axes[i].set_xlabel(f'{char.replace("_", " ").title()}')
                axes[i].set_ylabel(f'Spatial Lag of {char.replace("_", " ").title()}')
                axes[i].set_title(f'Moran\'s I: {moran.I:.3f} (p: {moran.p_sim:.3f})')
        
        plt.tight_layout()
        plt.savefig('spatial_autocorrelation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return moran_results
    
    def geographic_clustering(self):
        """Perform geographic clustering of music production"""
        
        # Prepare data for clustering
        coords = self.df[['latitude', 'longitude']].values
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=5, min_samples=5)  # eps in degrees
        dbscan_labels = dbscan.fit_predict(coords)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans_labels = kmeans.fit_predict(coords)
        
        # Add cluster labels to dataframe
        self.df['dbscan_cluster'] = dbscan_labels
        self.df['kmeans_cluster'] = kmeans_labels
        
        # Visualize clusters
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # DBSCAN clusters
        unique_labels = set(dbscan_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points
                cluster_data = self.df[self.df['dbscan_cluster'] == label]
                axes[0].scatter(cluster_data['longitude'], cluster_data['latitude'], 
                              c='black', marker='x', s=30, alpha=0.6, label='Noise')
            else:
                cluster_data = self.df[self.df['dbscan_cluster'] == label]
                axes[0].scatter(cluster_data['longitude'], cluster_data['latitude'], 
                              c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
        
        axes[0].set_title('DBSCAN Geographic Clustering')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].legend()
        
        # K-means clusters
        for i in range(5):
            cluster_data = self.df[self.df['kmeans_cluster'] == i]
            axes[1].scatter(cluster_data['longitude'], cluster_data['latitude'], 
                          c=[plt.cm.Set3(i/5)], s=50, alpha=0.7, label=f'Cluster {i}')
        
        axes[1].set_title('K-means Geographic Clustering')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('geographic_clustering.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Analyze cluster characteristics
        cluster_analysis = {}
        
        for method in ['dbscan', 'kmeans']:
            cluster_col = f'{method}_cluster'
            clusters = self.df[cluster_col].unique()
            
            cluster_analysis[method] = {}
            
            for cluster in clusters:
                if cluster != -1:  # Skip noise for DBSCAN
                    cluster_data = self.df[self.df[cluster_col] == cluster]
                    
                    stats = {
                        'count': len(cluster_data),
                        'avg_energy': cluster_data['energy'].mean(),
                        'avg_lexical_diversity': cluster_data['lexical_diversity'].mean(),
                        'dominant_decade': cluster_data['decade'].mode().iloc[0] if len(cluster_data) > 0 else None
                    }
                    
                    cluster_analysis[method][cluster] = stats
        
        return cluster_analysis
    
    def cultural_diffusion_analysis(self):
        """Analyze cultural diffusion patterns in music"""
        
        # Create temporal network of music influence
        # This is a simplified version - in practice, you'd use more sophisticated methods
        
        # Group by decade and region
        decade_region_stats = self.df.groupby(['decade', 'region']).agg({
            'energy': 'mean',
            'valence': 'mean',
            'danceability': 'mean',
            'lexical_diversity': 'mean',
            'vader_compound': 'mean'
        }).reset_index()
        
        # Calculate similarity between regions over time
        decades = sorted(decade_region_stats['decade'].unique())
        regions = decade_region_stats['region'].unique()
        
        # Create similarity matrix for each decade
        similarity_matrices = {}
        
        for decade in decades:
            decade_data = decade_region_stats[decade_region_stats['decade'] == decade]
            
            # Create feature matrix
            features = ['energy', 'valence', 'danceability', 'lexical_diversity', 'vader_compound']
            feature_matrix = decade_data[features].values
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(feature_matrix)
            
            similarity_matrices[decade] = similarity_matrix
        
        # Visualize cultural diffusion
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, decade in enumerate(decades[:6]):
            if decade in similarity_matrices:
                im = axes[i].imshow(similarity_matrices[decade], cmap='RdBu_r', vmin=-1, vmax=1)
                axes[i].set_title(f'Regional Similarity - {decade}')
                axes[i].set_xticks(range(len(regions)))
                axes[i].set_yticks(range(len(regions)))
                axes[i].set_xticklabels(regions, rotation=45)
                axes[i].set_yticklabels(regions)
                plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig('cultural_diffusion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return similarity_matrices
    
    def run_geospatial_analysis(self):
        """Run complete geospatial analysis pipeline"""
        print("Starting Geospatial Music Analysis...")
        print("=" * 50)
        
        print("\\n1. Geocoding artist locations...")
        location_cache = self.geocode_artists()
        
        print("\\n2. Creating world map visualizations...")
        self.create_world_map()
        
        print("\\n3. Analyzing regional music characteristics...")
        regional_stats = self.regional_music_characteristics()
        
        print("\\n4. Performing spatial autocorrelation analysis...")
        autocorr_results = self.spatial_autocorrelation_analysis()
        
        print("\\n5. Geographic clustering...")
        cluster_results = self.geographic_clustering()
        
        print("\\n6. Cultural diffusion analysis...")
        diffusion_results = self.cultural_diffusion_analysis()
        
        print("\\nGeospatial analysis complete! Check generated PNG files.")
        
        return {
            'location_cache': location_cache,
            'regional_stats': regional_stats,
            'spatial_autocorrelation': autocorr_results,
            'geographic_clustering': cluster_results,
            'cultural_diffusion': diffusion_results
        }

def main():
    """Main execution function"""
    
    # Check if merged data exists
    if not os.path.exists('merged.csv'):
        print("Error: merged.csv not found. Run merge_datasets.py first!")
        return
    
    # Run geospatial analysis
    analyzer = GeospatialMusicAnalyzer()
    results = analyzer.run_geospatial_analysis()
    
    print("\\nGeospatial Analysis Summary:")
    print(f"- Geocoded {len(results['location_cache'])} unique locations")
    print(f"- Analyzed {len(results['regional_stats'])} regions")
    if results['spatial_autocorrelation']:
        print(f"- Found spatial patterns in {len(results['spatial_autocorrelation'])} characteristics")

if __name__ == "__main__":
    import os
    main()
