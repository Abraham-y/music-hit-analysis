"""
Advanced Audio Feature Analysis
Implements sophisticated audio analysis techniques including:
- Gaussian Mixture Models for audio feature clustering
- Dynamic Time Warping for tempo analysis
- Audio feature embedding with autoencoders
- Causal inference for audio feature impact on chart success
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import networkx as nx
from dtw import dtw
from pyinform import mutual_info
import warnings
warnings.filterwarnings('ignore')

class AdvancedAudioAnalyzer:
    def __init__(self, audio_file='audio_clean.csv'):
        """Initialize with advanced audio analysis capabilities"""
        self.df = pd.read_csv(audio_file)
        self.audio_features = [
            'danceability', 'energy', 'valence', 'tempo', 
            'acousticness', 'instrumentalness', 'speechiness', 'loudness'
        ]
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data with advanced feature engineering"""
        # Add decade column
        self.df['decade'] = (self.df['year'] // 10) * 10
        self.df['decade'] = self.df['decade'].astype(str) + 's'
        
        # Advanced feature engineering
        self.engineer_audio_features()
        
        # Normalize features
        scaler = StandardScaler()
        self.df[self.audio_features] = scaler.fit_transform(self.df[self.audio_features])
        
        print(f"Loaded {len(self.df)} songs with advanced audio features")
    
    def engineer_audio_features(self):
        """Create sophisticated audio feature combinations"""
        # Energy-Valence ratio (emotional intensity)
        self.df['energy_valence_ratio'] = self.df['energy'] / (self.df['valence'] + 0.001)
        
        # Acoustic-Energy contrast (production style)
        self.df['acoustic_energy_contrast'] = abs(self.df['acousticness'] - self.df['energy'])
        
        # Danceability-Speechiness interaction (vocal dance music)
        self.df['dance_speech_interaction'] = self.df['danceability'] * self.df['speechiness']
        
        # Tempo normalized by energy (fast but chill vs fast and intense)
        self.df['tempo_energy_normalized'] = self.df['tempo'] * self.df['energy']
        
        # Loudness-Valence correlation proxy
        self.df['loudness_valence_proxy'] = self.df['loudness'] * self.df['valence']
        
        # Instrumentalness-Speechiness inverse relationship
        self.df['vocal_instrumental_balance'] = 1 - abs(self.df['instrumentalness'] - (1 - self.df['speechiness']))
        
        # Add engineered features to the list
        self.engineered_features = [
            'energy_valence_ratio', 'acoustic_energy_contrast', 'dance_speech_interaction',
            'tempo_energy_normalized', 'loudness_valence_proxy', 'vocal_instrumental_balance'
        ]
        
        self.all_features = self.audio_features + self.engineered_features
    
    def gaussian_mixture_clustering(self, n_components=8):
        """Apply Gaussian Mixture Models for probabilistic clustering"""
        
        # Prepare data
        X = self.df[self.all_features].values
        
        # Try different covariance types
        covariance_types = ['full', 'tied', 'diag', 'spherical']
        results = {}
        
        for cov_type in covariance_types:
            gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, 
                                random_state=42, n_init=10)
            labels = gmm.fit_predict(X)
            
            # Calculate metrics
            silhouette = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            
            results[cov_type] = {
                'model': gmm,
                'labels': labels,
                'silhouette': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'bic': bic,
                'aic': aic
            }
        
        # Select best model based on BIC
        best_cov_type = min(results.keys(), key=lambda x: results[x]['bic'])
        best_model = results[best_cov_type]
        
        # Add cluster labels to dataframe
        self.df['gmm_cluster'] = best_model['labels']
        self.df['gmm_probability'] = np.max(best_model['model'].predict_proba(X), axis=1)
        
        print(f"Best GMM: {best_cov_type} covariance")
        print(f"Silhouette score: {best_model['silhouette']:.3f}")
        print(f"BIC: {best_model['bic']:.1f}")
        
        return best_model, results
    
    def hierarchical_tempo_analysis(self):
        """Analyze tempo patterns using hierarchical clustering and DTW"""
        
        # Extract tempo data by decade
        tempo_by_decade = {}
        for decade in sorted(self.df['decade'].unique()):
            decade_tempos = self.df[self.df['decade'] == decade]['tempo'].values
            tempo_by_decade[decade] = np.sort(decade_tempos)
        
        # Calculate tempo distributions
        tempo_distributions = {}
        for decade, tempos in tempo_by_decade.items():
            # Create histogram for each decade
            hist, bin_edges = np.histogram(tempos, bins=50, density=True)
            tempo_distributions[decade] = (hist, bin_edges)
        
        # Calculate DTW distances between decade tempo distributions
        decades = list(tempo_distributions.keys())
        dtw_matrix = np.zeros((len(decades), len(decades)))
        
        for i, decade1 in enumerate(decades):
            for j, decade2 in enumerate(decades):
                if i <= j:
                    hist1, bins1 = tempo_distributions[decade1]
                    hist2, bins2 = tempo_distributions[decade2]
                    
                    # DTW distance
                    dist, _, _, _ = dtw(hist1, hist2, dist=lambda x, y: abs(x-y))
                    dtw_matrix[i, j] = dist
                    dtw_matrix[j, i] = dist
        
        # Create hierarchical clustering
        distance_matrix = squareform(dtw_matrix)
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # Plot dendrogram
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=decades, leaf_rotation=45)
        plt.title('Hierarchical Clustering of Tempo Distributions (DTW Distance)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Decade')
        plt.ylabel('DTW Distance')
        plt.tight_layout()
        plt.savefig('tempo_dtw_dendrogram.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return dtw_matrix, linkage_matrix
    
    def autoencoder_embedding(self, encoding_dim=3):
        """Create audio feature embeddings using autoencoders"""
        
        from sklearn.neural_network import MLPRegressor
        
        # Prepare data
        X = self.df[self.all_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create autoencoder architecture
        # Encoder: input -> encoding_dim
        # Decoder: encoding_dim -> input
        encoder = MLPRegressor(
            hidden_layer_sizes=[encoding_dim],
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        
        # Train autoencoder (input reconstruction)
        encoder.fit(X_scaled, X_scaled)
        
        # Get embeddings
        embeddings = encoder.predict(X_scaled)
        
        # Add embeddings to dataframe
        for i in range(encoding_dim):
            self.df[f'autoencoder_dim_{i+1}'] = embeddings[:, i]
        
        # Visualize embeddings
        if encoding_dim >= 2:
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                                c=self.df['year'], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Year')
            plt.xlabel('Autoencoder Dimension 1')
            plt.ylabel('Autoencoder Dimension 2')
            plt.title('Audio Feature Autoencoder Embeddings', fontsize=14, fontweight='bold')
            plt.savefig('audio_autoencoder_embeddings.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return embeddings, encoder
    
    def causal_inference_analysis(self):
        """Apply causal inference techniques to analyze audio feature impact"""
        
        # Define treatment: high energy vs low energy songs
        median_energy = self.df['energy'].median()
        self.df['high_energy_treatment'] = (self.df['energy'] > median_energy).astype(int)
        
        # Define outcome: chart success (top 10)
        self.df['chart_success'] = (self.df['chart_position'] <= 10).astype(int)
        
        # Propensity score matching
        from sklearn.linear_model import LogisticRegression
        
        # Features for propensity score
        propensity_features = ['danceability', 'valence', 'tempo', 'acousticness', 
                              'instrumentalness', 'speechiness', 'loudness']
        
        # Calculate propensity scores
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(self.df[propensity_features], self.df['high_energy_treatment'])
        propensity_scores = ps_model.predict_proba(self.df[propensity_features])[:, 1]
        
        self.df['propensity_score'] = propensity_scores
        
        # Perform matching
        treated = self.df[self.df['high_energy_treatment'] == 1]
        control = self.df[self.df['high_energy_treatment'] == 0]
        
        # Find matches for each treated song
        matched_pairs = []
        for idx, treated_song in treated.iterrows():
            # Find control song with closest propensity score
            control['distance'] = abs(control['propensity_score'] - treated_song['propensity_score'])
            best_match = control.loc[control['distance'].idxmin()]
            matched_pairs.append((treated_song, best_match))
        
        # Calculate Average Treatment Effect (ATE)
        treatment_effects = []
        for treated_song, control_song in matched_pairs:
            effect = treated_song['chart_success'] - control_song['chart_success']
            treatment_effects.append(effect)
        
        ate = np.mean(treatment_effects)
        
        # Bootstrap confidence intervals
        bootstrap_ates = []
        for _ in range(1000):
            sample_effects = np.random.choice(treatment_effects, size=len(treatment_effects), replace=True)
            bootstrap_ates.append(np.mean(sample_effects))
        
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)
        
        # Visualize results
        plt.figure(figsize=(12, 6))
        
        # Propensity score distributions
        plt.subplot(1, 2, 1)
        plt.hist(treated['propensity_score'], alpha=0.5, label='High Energy', bins=30)
        plt.hist(control['propensity_score'], alpha=0.5, label='Low Energy', bins=30)
        plt.xlabel('Propensity Score')
        plt.ylabel('Frequency')
        plt.title('Propensity Score Distributions')
        plt.legend()
        
        # Treatment effect
        plt.subplot(1, 2, 2)
        plt.bar(['ATE'], [ate], yerr=[[ate-ci_lower], [ci_upper-ate]], 
               capsize=5, alpha=0.7, color='coral')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.ylabel('Average Treatment Effect')
        plt.title(f'Causal Effect of High Energy on Chart Success\\nATE: {ate:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])')
        
        plt.tight_layout()
        plt.savefig('causal_inference_energy_effect.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return ate, (ci_lower, ci_upper), matched_pairs
    
    def network_analysis(self):
        """Create similarity network of songs based on audio features"""
        
        # Calculate pairwise similarity matrix
        X = self.df[self.all_features].values
        
        # Use cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(X)
        
        # Create network graph (only include strong similarities)
        threshold = np.percentile(similarity_matrix, 95)  # Top 5% similarities
        G = nx.Graph()
        
        # Add nodes
        for idx, row in self.df.iterrows():
            G.add_node(idx, 
                      title=row['title'], 
                      artist=row['artist'], 
                      year=row['year'],
                      decade=row['decade'])
        
        # Add edges for high similarities
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                if similarity_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        # Analyze network properties
        print(f"Network stats:")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")
        
        # Community detection
        communities = nx.community.louvain_communities(G)
        print(f"Number of communities: {len(communities)}")
        
        # Add community labels to dataframe
        self.df['network_community'] = -1
        for comm_id, community in enumerate(communities):
            for node in community:
                self.df.loc[node, 'network_community'] = comm_id
        
        # Visualize network (sample for performance)
        if len(G) > 500:
            # Sample nodes for visualization
            sample_nodes = np.random.choice(list(G.nodes()), size=500, replace=False)
            G_vis = G.subgraph(sample_nodes)
        else:
            G_vis = G
        
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G_vis, k=1, iterations=50)
        
        # Color by decade
        decade_colors = {}
        decades = sorted(self.df['decade'].unique())
        colors = plt.cm.Set3(np.linspace(0, 1, len(decades)))
        decade_colors = {decade: colors[i] for i, decade in enumerate(decades)}
        
        node_colors = [decade_colors[G_vis.nodes[node]['decade']] for node in G_vis.nodes()]
        
        nx.draw(G_vis, pos, node_color=node_colors, node_size=50, 
                alpha=0.7, with_labels=False, edge_color='gray', width=0.5)
        
        plt.title('Audio Feature Similarity Network\\n(Colored by Decade)', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.savefig('audio_similarity_network.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return G, communities
    
    def advanced_feature_importance(self):
        """Use SHAP values and permutation importance for feature analysis"""
        
        try:
            import shap
            from sklearn.inspection import permutation_importance
        except ImportError:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        # Prepare data
        X = self.df[self.all_features]
        y = self.df['chart_position'] <= 10  # Top 10 binary
        
        # Train a model
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Permutation importance
        perm_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=True)
        
        axes[0, 0].barh(perm_importances['feature'], perm_importances['importance'], 
                        xerr=perm_importances['std'], alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Permutation Feature Importance')
        axes[0, 0].set_xlabel('Importance')
        
        # SHAP summary plot
        shap.summary_plot(shap_values[1], X, plot_type="bar", ax=axes[0, 1])
        axes[0, 1].set_title('SHAP Feature Importance')
        
        # SHAP dependence plots for top features
        top_features = perm_importances['feature'].tail(2).values
        for i, feature in enumerate(top_features):
            shap.dependence_plot(feature, shap_values[1], X, ax=axes[1, i])
        
        plt.tight_layout()
        plt.savefig('advanced_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return perm_importances, shap_values
    
    def run_advanced_analysis(self):
        """Run complete advanced analysis pipeline"""
        print("Starting Advanced Audio Analysis...")
        print("=" * 50)
        
        print("\\n1. Gaussian Mixture Model clustering...")
        gmm_results, all_gmm_results = self.gaussian_mixture_clustering()
        
        print("\\n2. Hierarchical tempo analysis with DTW...")
        dtw_matrix, linkage_matrix = self.hierarchical_tempo_analysis()
        
        print("\\n3. Autoencoder feature embeddings...")
        embeddings, encoder = self.autoencoder_embedding()
        
        print("\\n4. Causal inference analysis...")
        ate, ci, matched_pairs = self.causal_inference_analysis()
        
        print("\\n5. Network similarity analysis...")
        network, communities = self.network_analysis()
        
        print("\\n6. Advanced feature importance...")
        perm_importance, shap_values = self.advanced_feature_importance()
        
        print("\\nAdvanced analysis complete! Check generated PNG files.")
        
        return {
            'gmm_results': gmm_results,
            'dtw_analysis': (dtw_matrix, linkage_matrix),
            'autoencoder': (embeddings, encoder),
            'causal_inference': (ate, ci, matched_pairs),
            'network_analysis': (network, communities),
            'feature_importance': (perm_importance, shap_values)
        }

def main():
    """Main execution function"""
    
    # Check if audio data exists
    if not os.path.exists('audio_clean.csv'):
        print("Error: audio_clean.csv not found. Run spotify_audio_features.py first!")
        return
    
    # Run advanced analysis
    analyzer = AdvancedAudioAnalyzer()
    results = analyzer.run_advanced_analysis()
    
    print("\\nAdvanced Analysis Summary:")
    print(f"- GMM clustering identified {len(np.unique(analyzer.df['gmm_cluster']))} distinct audio profiles")
    print(f"- Causal effect of high energy: {results['causal_inference'][0]:.3f}")
    print(f"- Network contains {results['network_analysis'][0].number_of_nodes()} nodes and {results['network_analysis'][0].number_of_edges()} edges")
    print(f"- Autoencoder reduced {len(analyzer.all_features)} features to 3 dimensions")

if __name__ == "__main__":
    import os
    from scipy.cluster.hierarchy import linkage
    main()
