"""
NLP Lyrics Analysis
Analyzes lyrics for linguistic patterns, sentiment, and clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.manifold import TSNE
import umap
from gensim.models import Word2Vec
from textblob import TextBlob
from collections import Counter
import re
from tqdm import tqdm

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NLPAnalyzer:
    def __init__(self, lyrics_file='lyrics_clean.csv'):
        """Initialize with lyrics data"""
        self.df = pd.read_csv(lyrics_file)
        self.df = self.df.dropna(subset=['lyrics_clean'])
        self.prepare_data()
        
    
    def prepare_data(self):
        """Prepare data for analysis"""
        # Add decade column
        self.df['decade'] = (self.df['year'] // 10) * 10
        self.df['decade'] = self.df['decade'].astype(str) + 's'

        # Drop bad Genius scrapes: real songs are almost never over 800 words.
        before = len(self.df)
        self.df = self.df[self.df['word_count'] <= 800].copy()
        dropped = before - len(self.df)
        if dropped > 0:
            print(f"Dropped {dropped} likely bad scrapes (word_count > 800)")

        # Merge chart position from songs.csv so we can filter by top-N
        if 'chart_position' not in self.df.columns:
            try:
                songs = pd.read_csv('songs.csv')[['title', 'artist', 'year', 'chart_position']]
                self.df = self.df.merge(songs, on=['title', 'artist', 'year'], how='left')
            except FileNotFoundError:
                print("Warning: songs.csv not found — chart_position filtering unavailable")

        print(f"Loaded {len(self.df)} songs with lyrics")
        print(f"Decades covered: {', '.join(sorted(self.df['decade'].unique()))}")

    @property
    def lyric_stopwords(self):
        """Shared stopword list for all vectorizers: sklearn English + contraction
        artifacts + lyric filler + metadata leakage + Spanish filler."""
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return list(set(ENGLISH_STOP_WORDS) | {
            # Contraction artifacts (apostrophe stripped)
            'll', 've', 're', 'd', 't', 's', 'ain', 'don', 'won', 'can',
            'didn', 'doesn', 'isn', 'wasn', 'weren', 'wouldn', 'couldn', 'shouldn',
            # Lyric filler words
            'oh', 'ooh', 'ah', 'yeah', 'ya', 'na', 'la', 'da', 'hey', 'uh',
            'wanna', 'gonna', 'gotta', 'whoa', 'mmm', 'hmm', 'ha', 'baby',
            'like', 'just', 'got', 'know', 'let', 'say', 'said', 'come',
            'get', 'go', 'going', 'cause', 'cuz', 'make', 'take', 'want',
            'need', 'feel', 'way', 'right', 'left', 'little', 'bout',
            # Metadata leakage
            'feat', 'ft', 'remix', 'edit', 'version', 'radio',
            # Spanish filler
            'que', 'tu', 'mi', 'te', 'yo', 'es', 'de', 'en', 'el', 'lo',
        })

    def analyze_lexical_diversity(self):
        """Analyze lexical diversity trends over time"""
        # Calculate yearly statistics
        yearly_diversity = self.df.groupby('year')['lexical_diversity'].agg(['mean', 'std', 'count'])

        # Calculate decade statistics
        decade_diversity = self.df.groupby('decade')['lexical_diversity'].agg(['mean', 'std', 'count'])

        print("\nLexical Diversity by Decade:")
        print(decade_diversity.round(3).to_string())
        print("\nLexical Diversity by Year:")
        print(yearly_diversity.round(3).to_string())
        
        # Plot lexical diversity over time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Yearly trend with confidence bands
        years = yearly_diversity.index
        means = yearly_diversity['mean']
        stds = yearly_diversity['std']
        
        ax1.plot(years, means, linewidth=3, color='royalblue', label='Mean Lexical Diversity')
        ax1.fill_between(years, means - stds, means + stds, alpha=0.3, color='royalblue')
        
        # Add quadratic trend line (U-shape: simpler through 2000s, then streaming reversal)
        z = np.polyfit(years, means, 2)
        p = np.poly1d(z)
        years_smooth = np.linspace(years.min(), years.max(), 300)
        residuals = means - p(years)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((means - means.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        ax1.plot(years_smooth, p(years_smooth), "r--", alpha=0.8, linewidth=2,
                 label=f'Quadratic trend (R²={r_squared:.3f}, min ≈ {int(-z[1]/(2*z[0]))})')
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Lexical Diversity', fontsize=12)
        ax1.set_title('Lexical Diversity Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Decade comparison
        decades = decade_diversity.index
        means_decade = decade_diversity['mean']
        stds_decade = decade_diversity['std']
        
        bars = ax2.bar(decades, means_decade, yerr=stds_decade, 
                      capsize=5, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Decade', fontsize=12)
        ax2.set_ylabel('Mean Lexical Diversity', fontsize=12)
        ax2.set_title('Lexical Diversity by Decade', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means_decade, stds_decade):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('lexical_diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return yearly_diversity, decade_diversity
    
    def analyze_sentiment(self):
        """Analyze sentiment trends in lyrics"""
        # Calculate sentiment for all lyrics
        sentiments = []
        
        for lyrics in tqdm(self.df['lyrics_clean'], desc="Analyzing sentiment"):
            blob = TextBlob(lyrics)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            sentiments.append({
                'vader_compound': polarity,  # use textblob polarity as proxy
                'vader_positive': max(polarity, 0),
                'vader_negative': abs(min(polarity, 0)),
                'vader_neutral': 1 - abs(polarity),
                'textblob_polarity': polarity,
                'textblob_subjectivity': subjectivity
            })
        
        # Add to dataframe
        sentiment_df = pd.DataFrame(sentiments)
        self.df = pd.concat([self.df.reset_index(drop=True), sentiment_df], axis=1)
        self.df = self.df.loc[:, ~self.df.columns.duplicated(keep='last')]
        
        # Analyze trends
        yearly_sentiment = self.df.groupby('year')[['vader_compound', 'textblob_polarity']].mean()
        
        # Plot sentiment trends
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sentiment over time
        axes[0, 0].plot(yearly_sentiment.index, yearly_sentiment['vader_compound'],
                       linewidth=3, color='darkgreen')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Sentiment Score')
        axes[0, 0].set_title('Sentiment Over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # TextBlob subjectivity over time
        axes[0, 1].plot(yearly_sentiment.index, yearly_sentiment['textblob_polarity'],
                       linewidth=3, color='darkorange')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('TextBlob Polarity')
        axes[0, 1].set_title('TextBlob Sentiment Over Time')
        axes[0, 1].grid(True, alpha=0.3)

        # Sentiment distribution by decade
        decade_sentiment = self.df.groupby('decade')['vader_compound'].mean()
        bars = axes[1, 0].bar(decade_sentiment.index, decade_sentiment.values,
                             alpha=0.7, color='mediumpurple')
        axes[1, 0].set_xlabel('Decade')
        axes[1, 0].set_ylabel('Mean Sentiment')
        axes[1, 0].set_title('Average Sentiment by Decade')
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, decade_sentiment.values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom')

        # Sentiment vs Lexical Diversity scatter
        axes[1, 1].scatter(self.df['lexical_diversity'], self.df['vader_compound'],
                          alpha=0.5, s=30)
        axes[1, 1].set_xlabel('Lexical Diversity')
        axes[1, 1].set_ylabel('Sentiment Score')
        axes[1, 1].set_title('Sentiment vs Lexical Diversity')

        # Add correlation
        corr = self.df['lexical_diversity'].corr(self.df['vader_compound'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                       transform=axes[1, 1].transAxes, va='top')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Save enriched dataframe back so merge_datasets.py can access sentiment columns
        self.df.to_csv('lyrics_clean.csv', index=False)

        return yearly_sentiment
    
    def analyze_vocabulary_drift(self):
        """Analyze vocabulary changes over decades using TF-IDF"""
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=self.lyric_stopwords,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8
        )
        
        # Fit and transform all lyrics
        self.vectorizer = vectorizer
        tfidf_matrix = vectorizer.fit_transform(self.df['lyrics_clean'])
        feature_names = vectorizer.get_feature_names_out()
        
        # Create DataFrame with TF-IDF scores
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        tfidf_df['decade'] = self.df['decade'].values
        
        # Calculate average TF-IDF scores by decade
        decade_tfidf = tfidf_df.groupby('decade').mean()
        
        # Get top words for each decade
        top_words_per_decade = {}
        for decade in decade_tfidf.index:
            decade_scores = decade_tfidf.loc[decade]
            top_words = decade_scores.nlargest(20)
            top_words_per_decade[decade] = top_words
        
        # Create heatmap of top words
        all_top_words = set()
        for words in top_words_per_decade.values():
            all_top_words.update(words.index)
        
        # Prepare heatmap data
        heatmap_data = []
        for decade in sorted(top_words_per_decade.keys()):
            decade_words = top_words_per_decade[decade]
            row = []
            for word in sorted(all_top_words):
                row.append(decade_words.get(word, 0))
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=sorted(top_words_per_decade.keys()),
                                 columns=sorted(all_top_words))
        
        # Plot heatmap
        plt.figure(figsize=(16, 10))
        sns.heatmap(heatmap_df, cmap='YlOrRd', annot=False, 
                   cbar_kws={'label': 'Average TF-IDF Score'})
        plt.title('Vocabulary Drift: Top Words by Decade', fontsize=16, fontweight='bold')
        plt.xlabel('Words')
        plt.ylabel('Decade')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('vocabulary_drift_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return top_words_per_decade, vectorizer, tfidf_matrix
    
    def perform_lyrical_clustering(self, tfidf_matrix, n_clusters=8):
        """Perform KMeans clustering on lyrics"""
        # Reduce dimensionality with SVD first
        svd = TruncatedSVD(n_components=50, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_matrix)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_reduced)
        
        # Add cluster labels to dataframe
        self.df['lyrical_cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(n_clusters):
            cluster_songs = self.df[self.df['lyrical_cluster'] == cluster_id]
            
            # Get top words for this cluster
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_tfidf = tfidf_matrix[cluster_indices]
            mean_tfidf = np.mean(cluster_tfidf.toarray(), axis=0)
            
            # Get top 10 words
            feature_names = self.vectorizer.get_feature_names_out()
            top_word_indices = np.argsort(mean_tfidf)[-10:]
            top_words = [feature_names[i] for i in top_word_indices]
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_songs),
                'top_words': top_words,
                'avg_lexical_diversity': cluster_songs['lexical_diversity'].mean(),
                'avg_sentiment': cluster_songs['vader_compound'].mean(),
                'dominant_decade': cluster_songs['decade'].mode().iloc[0] if len(cluster_songs) > 0 else None
            }
        
        return cluster_analysis, tfidf_reduced
    
    def create_umap_visualization(self, tfidf_reduced=None):
        """UMAP visualization of lyrical space using only rare/distinctive words, colored by decade.

        Builds its own TF-IDF restricted to high-IDF (rare) words — words that appear
        in fewer than 20% of songs. Common words that appear everywhere add noise and
        blur decade separation.
        """
        from sklearn.decomposition import TruncatedSVD
        from scipy.spatial import ConvexHull

        # Build rare-word TF-IDF: max_df=0.2 keeps only words present in <20% of songs
        rare_vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            min_df=5,       # must appear in at least 5 songs (not a typo/one-off)
            max_df=0.2,     # must appear in fewer than 20% of songs (rare = distinctive)
            ngram_range=(1, 1)
        )
        tfidf_rare = rare_vectorizer.fit_transform(self.df['lyrics_clean'])

        # Print most distinctive words to confirm quality
        idf_scores = rare_vectorizer.idf_
        feature_names = rare_vectorizer.get_feature_names_out()
        top_rare = [feature_names[i] for i in np.argsort(idf_scores)[-20:][::-1]]
        print(f"Top rare/distinctive words used for UMAP: {', '.join(top_rare)}")
        print(f"Vocabulary size (rare words only): {tfidf_rare.shape[1]}")

        # Reduce dimensionality with SVD before UMAP (UMAP works poorly on sparse high-dim data)
        n_components = min(50, tfidf_rare.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf_dense = svd.fit_transform(tfidf_rare)

        # UMAP to 2D
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(tfidf_dense)

        # Plot colored by decade with convex hulls
        decades = sorted(self.df['decade'].unique())
        colors = sns.color_palette('husl', len(decades))

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Left: scatter only
        for i, decade in enumerate(decades):
            mask = self.df['decade'].values == decade
            axes[0].scatter(embedding[mask, 0], embedding[mask, 1],
                            c=[colors[i]], label=decade, alpha=0.5, s=30)
        axes[0].set_xlabel('UMAP 1', fontsize=12)
        axes[0].set_ylabel('UMAP 2', fontsize=12)
        axes[0].set_title('Lyrical Space by Decade\n(rare/distinctive words only)',
                          fontsize=13, fontweight='bold')
        axes[0].legend(title='Decade', fontsize=9)
        axes[0].grid(True, alpha=0.2)

        # Right: scatter + convex hulls to show decade territory
        for i, decade in enumerate(decades):
            mask = self.df['decade'].values == decade
            pts = embedding[mask]
            axes[1].scatter(pts[:, 0], pts[:, 1],
                            c=[colors[i]], alpha=0.3, s=20)
            if pts.shape[0] >= 3:
                try:
                    hull = ConvexHull(pts)
                    for simplex in hull.simplices:
                        axes[1].plot(pts[simplex, 0], pts[simplex, 1],
                                     color=colors[i], linewidth=2, alpha=0.8)
                    # Label centroid
                    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                    axes[1].text(cx, cy, decade, fontsize=9, fontweight='bold',
                                 ha='center', va='center',
                                 color=colors[i],
                                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
                except Exception:
                    pass

        axes[1].set_xlabel('UMAP 1', fontsize=12)
        axes[1].set_ylabel('UMAP 2', fontsize=12)
        axes[1].set_title('Lyrical Territory by Decade\n(convex hulls — overlap = similar vocabulary)',
                          fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.2)

        plt.suptitle('UMAP of Song Lyrics (Rare & Distinctive Words Only)',
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig('umap_lyrical_space.png', dpi=300, bbox_inches='tight')
        plt.show()

        return embedding
    
    def analyze_lda_topics(self, n_topics=6):
        """LDA topic modeling to show how lyrical themes shift by decade"""
        # Use count vectorizer (LDA needs raw counts, not TF-IDF)
        count_vec = CountVectorizer(max_features=3000, stop_words=self.lyric_stopwords,
                                    min_df=5, max_df=0.8)
        count_matrix = count_vec.fit_transform(self.df['lyrics_clean'])
        feature_names = count_vec.get_feature_names_out()

        # Fit LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                        max_iter=20, learning_method='online')
        doc_topics = lda.fit_transform(count_matrix)

        # Get top words per topic
        topic_labels = []
        for i, topic in enumerate(lda.components_):
            top_words = [feature_names[j] for j in topic.argsort()[-8:][::-1]]
            topic_labels.append(f"Topic {i+1}: {', '.join(top_words[:4])}")

        # Add dominant topic to df
        self.df['dominant_topic'] = doc_topics.argmax(axis=1)

        # Plot: topic prevalence by decade
        topic_by_decade = pd.DataFrame(doc_topics, columns=[f'Topic {i+1}' for i in range(n_topics)])
        topic_by_decade['decade'] = self.df['decade'].values
        decade_topics = topic_by_decade.groupby('decade').mean()

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # Stacked bar: topic mix per decade
        decade_topics.plot(kind='bar', stacked=True, ax=axes[0],
                           colormap='tab10', alpha=0.85)
        axes[0].set_xlabel('Decade', fontsize=12)
        axes[0].set_ylabel('Mean Topic Proportion', fontsize=12)
        axes[0].set_title('Lyrical Theme Mix by Decade', fontsize=14, fontweight='bold')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].tick_params(axis='x', rotation=45)

        # Heatmap of topic proportions
        sns.heatmap(decade_topics.T, cmap='YlOrRd', annot=True, fmt='.2f',
                    ax=axes[1], cbar_kws={'label': 'Mean Proportion'})
        axes[1].set_title('Topic Heatmap by Decade', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Decade')
        axes[1].set_ylabel('Topic')

        plt.tight_layout()
        plt.savefig('lda_topic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nTop words per topic:")
        for label in topic_labels:
            print(f"  {label}")

        return lda, doc_topics, topic_labels

    def analyze_word2vec_drift(self, target_words=None):
        """Word2Vec semantic drift: train per decade and track how word context shifts"""
        if target_words is None:
            target_words = ['love', 'baby', 'heart', 'girl', 'night', 'time', 'feel']

        # Tokenize lyrics
        self.df['tokens'] = self.df['lyrics_clean'].apply(
            lambda x: [w for w in x.lower().split() if len(w) > 2])

        decades = sorted(self.df['decade'].unique())
        decade_models = {}

        for decade in decades:
            decade_lyrics = self.df[self.df['decade'] == decade]['tokens'].tolist()
            if len(decade_lyrics) < 10:
                continue
            model = Word2Vec(sentences=decade_lyrics, vector_size=100, window=5,
                             min_count=3, workers=4, epochs=20, seed=42)
            decade_models[decade] = model

        # For each target word, find most similar words per decade
        fig, axes = plt.subplots(len(target_words), 1,
                                 figsize=(14, 3 * len(target_words)))
        if len(target_words) == 1:
            axes = [axes]

        for ax, word in zip(axes, target_words):
            decades_present = []
            top_neighbors = []
            for decade, model in decade_models.items():
                if word in model.wv:
                    similar = model.wv.most_similar(word, topn=5)
                    decades_present.append(decade)
                    top_neighbors.append([w for w, _ in similar])

            if not decades_present:
                ax.axis('off')
                ax.set_title(f'"{word}" not found', fontsize=10)
                continue

            # Display as table-style text
            cell_text = [[', '.join(neighbors)] for neighbors in top_neighbors]
            ax.axis('off')
            table = ax.table(cellText=cell_text,
                             rowLabels=decades_present,
                             colLabels=[f'Top 5 words near "{word}"'],
                             loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.4)
            ax.set_title(f'Semantic neighbors of "{word}" by decade',
                         fontsize=11, fontweight='bold', pad=2)

        plt.suptitle('Word2Vec Semantic Drift: How Word Context Shifts Over Decades',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig('word2vec_semantic_drift.png', dpi=300, bbox_inches='tight')
        plt.show()

        return decade_models

    def analyze_repetition_ratio(self):
        """Analyze lyric repetition (1 - lexical_diversity) over time — modern pop is more repetitive"""
        self.df['repetition_ratio'] = 1 - self.df['lexical_diversity']

        yearly_rep = self.df.groupby('year')['repetition_ratio'].agg(['mean', 'std'])
        decade_rep = self.df.groupby('decade')['repetition_ratio'].mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        years = yearly_rep.index
        means = yearly_rep['mean']
        stds = yearly_rep['std']

        ax1.plot(years, means, linewidth=3, color='darkorange', label='Mean Repetition')
        ax1.fill_between(years, means - stds, means + stds, alpha=0.2, color='darkorange')

        z = np.polyfit(years, means, 2)
        p = np.poly1d(z)
        years_smooth = np.linspace(years.min(), years.max(), 300)
        ax1.plot(years_smooth, p(years_smooth), 'r--', linewidth=2, alpha=0.8,
                 label=f'Quadratic trend (peak ≈ {int(-z[1] / (2 * z[0]))})')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Repetition Ratio', fontsize=12)
        ax1.set_title('Lyric Repetition Over Time\n(Higher = more words are repeated)',
                      fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        colors = sns.color_palette('husl', len(decade_rep))
        bars = ax2.bar(decade_rep.index, decade_rep.values, color=colors, alpha=0.8)
        ax2.set_xlabel('Decade', fontsize=12)
        ax2.set_ylabel('Mean Repetition Ratio', fontsize=12)
        ax2.set_title('Lyric Repetition by Decade\n(Higher = more chorus/hook repetition)',
                      fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        for bar, val in zip(bars, decade_rep.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f'{val:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('lyrics_repetition_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nRepetition Ratio by Decade:")
        print(decade_rep.round(3).to_string())
        return yearly_rep, decade_rep

    def analyze_song_length(self):
        """Analyze song length (word count) over time — streaming shortening effect"""
        from scipy import stats as scipy_stats

        yearly_wc = self.df.groupby('year')['word_count'].agg(['mean', 'std', 'median'])
        decade_wc = self.df.groupby('decade')['word_count'].agg(['mean', 'std', 'median'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        years = yearly_wc.index.values
        means = yearly_wc['mean'].values
        stds = yearly_wc['std'].values

        axes[0].plot(years, means, linewidth=3, color='teal', label='Mean word count')
        axes[0].fill_between(years, means - stds, means + stds, alpha=0.2, color='teal')

        slope, intercept, r_value, p_value, _ = scipy_stats.linregress(years, means)
        axes[0].plot(years, slope * years + intercept, 'r--', linewidth=2,
                     label=f'Trend (slope={slope:.1f}/yr, p={p_value:.3f})')
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Word Count', fontsize=12)
        axes[0].set_title('Song Length (Words) Over Time\n(Streaming era: songs getting shorter?)',
                          fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        bars = axes[1].bar(decade_wc.index, decade_wc['mean'], yerr=decade_wc['std'],
                           capsize=5, alpha=0.7, color='teal')
        axes[1].set_xlabel('Decade', fontsize=12)
        axes[1].set_ylabel('Mean Word Count', fontsize=12)
        axes[1].set_title('Song Length by Decade', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        for bar, mean in zip(bars, decade_wc['mean']):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                         f'{mean:.0f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('lyrics_song_length.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nWord Count by Decade:")
        print(decade_wc[['mean', 'median']].round(1).to_string())
        return yearly_wc, decade_wc

    def test_decade_significance(self):
        """Statistical significance tests for lexical diversity and repetition across decades.

        Tests used:
        - Kruskal-Wallis: non-parametric one-way ANOVA across all decades
        - Pairwise Mann-Whitney U: which specific decade pairs differ significantly
        - Bonferroni correction applied to control for multiple comparisons
        """
        from scipy import stats
        from itertools import combinations

        if 'repetition_ratio' not in self.df.columns:
            self.df['repetition_ratio'] = 1 - self.df['lexical_diversity']

        decades = sorted(self.df['decade'].unique())
        # Only test lexical_diversity — repetition_ratio = 1 - lexical_diversity so
        # Mann-Whitney U gives mathematically identical results for both.
        metrics = {
            'Lexical Diversity': 'lexical_diversity',
        }

        all_results = {}

        for metric_name, col in metrics.items():
            groups = [self.df[self.df['decade'] == d][col].dropna().values for d in decades]

            # Kruskal-Wallis across all decades
            kw_stat, kw_p = stats.kruskal(*groups)

            # Pairwise Mann-Whitney U with Bonferroni correction
            pairs = list(combinations(range(len(decades)), 2))
            n_comparisons = len(pairs)
            pairwise = {}
            for i, j in pairs:
                u_stat, p_val = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                p_bonf = min(p_val * n_comparisons, 1.0)
                pairwise[(decades[i], decades[j])] = {
                    'u_stat': u_stat,
                    'p_raw': p_val,
                    'p_bonferroni': p_bonf,
                    'significant': p_bonf < 0.05
                }

            all_results[metric_name] = {
                'kruskal_stat': kw_stat,
                'kruskal_p': kw_p,
                'pairwise': pairwise
            }

            print(f"\n{'='*55}")
            print(f"{metric_name}")
            print(f"{'='*55}")
            print(f"Kruskal-Wallis H={kw_stat:.2f}, p={kw_p:.4f} "
                  f"({'SIGNIFICANT' if kw_p < 0.05 else 'not significant'})")
            print(f"\nPairwise Mann-Whitney U (Bonferroni corrected, n={n_comparisons}):")
            print(f"{'Pair':<25} {'p (raw)':>10} {'p (Bonf.)':>12} {'Sig?':>6}")
            print("-" * 55)
            for (d1, d2), res in pairwise.items():
                sig = '***' if res['p_bonferroni'] < 0.001 else ('**' if res['p_bonferroni'] < 0.01 else ('*' if res['significant'] else ''))
                print(f"{d1} vs {d2:<10} {res['p_raw']:>10.4f} {res['p_bonferroni']:>12.4f} {sig:>6}")

        # Visualize: p-value heatmap
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]  # keep loop compatible

        for ax, (metric_name, col) in zip(axes, metrics.items()):
            p_matrix = pd.DataFrame(np.ones((len(decades), len(decades))),
                                    index=decades, columns=decades)
            for (d1, d2), res in all_results[metric_name]['pairwise'].items():
                p_matrix.loc[d1, d2] = res['p_bonferroni']
                p_matrix.loc[d2, d1] = res['p_bonferroni']

            # Mask the diagonal
            mask = np.eye(len(decades), dtype=bool)
            annot = p_matrix.map(lambda v: f'{v:.3f}' if v < 1.0 else '-')

            sns.heatmap(p_matrix, mask=mask, annot=annot, fmt='', cmap='RdYlGn_r',
                        vmin=0, vmax=0.1, ax=ax, linewidths=0.5,
                        cbar_kws={'label': 'Bonferroni p-value'})
            ax.set_title(
                f'{metric_name}: Pairwise Significance\n'
                f'(Kruskal-Wallis p={all_results[metric_name]["kruskal_p"]:.4f})',
                fontsize=12, fontweight='bold'
            )
            ax.set_xlabel('Decade')
            ax.set_ylabel('Decade')

            # Mark significant cells with asterisk overlay
            for i, d1 in enumerate(decades):
                for j, d2 in enumerate(decades):
                    if i != j:
                        p = p_matrix.loc[d1, d2]
                        if p < 0.05:
                            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                                       edgecolor='black', lw=2))

        plt.suptitle('Lexical Diversity: Pairwise Significance Across Decades\n'
                     '(Green = significant difference, black border = p<0.05 after Bonferroni)\n'
                     'Note: Repetition ratio is 1 − lexical diversity; results are identical.',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig('lyrics_significance_tests.png', dpi=300, bbox_inches='tight')
        plt.show()

        return all_results

    @staticmethod
    def _lz77_compressibility(text, window_size=32768, min_match=4, max_match=258):
        """LZ77 compressibility per the paper methodology.

        compsize(S) = |S| - sum(L - 3) for all matches with L >= 4
        compressibility = log(|S| / compsize(S))

        Higher value = more repetitive/compressible lyrics.
        """
        import math
        n = len(text)
        if n < min_match:
            return 0.0

        total_savings = 0
        i = 0

        while i < n:
            best_len = 0

            if i + min_match <= n:
                win_start = max(0, i - window_size)
                seed = text[i:i + min_match]
                search_from = win_start

                while True:
                    pos = text.find(seed, search_from, i)
                    if pos == -1:
                        break
                    dist = i - pos
                    length = min_match
                    while i + length < n and length < max_match:
                        if text[pos + (length % dist)] == text[i + length]:
                            length += 1
                        else:
                            break
                    if length > best_len:
                        best_len = length
                    search_from = pos + 1

            if best_len >= min_match:
                total_savings += best_len - 3
                i += best_len
            else:
                i += 1

        comp_size = max(n - total_savings, 1)
        return math.log(n / comp_size)

    def analyze_compressibility(self):
        """Compute LZ77 compressibility for each song and plot trends by decade.

        Based on the methodology in the paper: compressibility indexes structural
        repetition (whole choruses, multi-line hooks) rather than just word reuse.
        Higher = more repetitive/simpler.
        """
        if 'compressibility' not in self.df.columns:
            print("Computing LZ77 compressibility (this takes a few minutes)...")
            from tqdm import tqdm
            tqdm.pandas(desc="LZ77")
            self.df['compressibility'] = self.df['lyrics_clean'].progress_apply(
                self._lz77_compressibility
            )
            # Save back so merge_datasets.py can include it in merged.csv
            self.df.to_csv('lyrics_clean.csv', index=False)
            print("Saved compressibility to lyrics_clean.csv")

        yearly_comp = self.df.groupby('year')['compressibility'].agg(['mean', 'std'])
        decade_comp = self.df.groupby('decade')['compressibility'].agg(['mean', 'std', 'count'])

        print("\nLZ77 Compressibility by Decade (higher = more repetitive):")
        print(decade_comp.round(3).to_string())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Yearly trend
        years = yearly_comp.index
        means = yearly_comp['mean']
        stds = yearly_comp['std']
        axes[0].plot(years, means, linewidth=3, color='crimson', label='Mean Compressibility')
        axes[0].fill_between(years, means - stds, means + stds, alpha=0.2, color='crimson')

        # Quadratic trend
        z = np.polyfit(years, means, 2)
        p = np.poly1d(z)
        years_smooth = np.linspace(years.min(), years.max(), 300)
        residuals = means - p(years)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((means - means.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        peak_year = int(-z[1] / (2 * z[0]))
        axes[0].plot(years_smooth, p(years_smooth), 'k--', linewidth=2, alpha=0.8,
                     label=f'Quadratic trend (R²={r2:.3f}, peak ≈ {peak_year})')
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('LZ77 Compressibility', fontsize=12)
        axes[0].set_title('Lyric Compressibility Over Time\n(Higher = more structurally repetitive)',
                          fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # By decade
        colors = sns.color_palette('husl', len(decade_comp))
        bars = axes[1].bar(decade_comp.index, decade_comp['mean'],
                           yerr=decade_comp['std'], capsize=5, color=colors, alpha=0.8)
        axes[1].set_xlabel('Decade', fontsize=12)
        axes[1].set_ylabel('Mean LZ77 Compressibility', fontsize=12)
        axes[1].set_title('Lyric Compressibility by Decade\n(Higher = more chorus/hook repetition)',
                          fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        for bar, val in zip(bars, decade_comp['mean']):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                         f'{val:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('lyrics_compressibility.png', dpi=300, bbox_inches='tight')
        plt.show()

        return yearly_comp, decade_comp

    def spot_check_lyrics(self, n=20, seed=42):
        """Print a random sample of (title, artist, year, lyrics_preview) for manual QC.

        Use this to verify Genius scraped the right song rather than e.g. a book,
        an album tracklist, or the wrong artist's song.
        """
        sample = self.df.sample(n=n, random_state=seed)[
            ['title', 'artist', 'year', 'word_count', 'lyrics_clean']
        ].reset_index(drop=True)

        print(f"\n{'='*70}")
        print(f"LYRICS SPOT CHECK — {n} random songs")
        print(f"{'='*70}")
        for _, row in sample.iterrows():
            preview = ' '.join(str(row['lyrics_clean']).split()[:25])
            print(f"\n[{int(row['year'])}] {row['title']} — {row['artist']}  (words: {int(row['word_count'])})")
            print(f"  \"{preview}...\"")
        print(f"\n{'='*70}")
        print("Check: does each preview sound like the right song?")
        print("Red flags: book text, tracklist, foreign language (if unexpected),")
        print("           completely wrong artist/genre, very short nonsense text.")

    def run_top_n_comparison(self, top_ns=(10, 50, 100)):
        """Compare lexical diversity, compressibility, and sentiment across chart tiers.

        Runs the same decade-level analysis on top-10, top-50, top-100 (full Hot 100)
        side by side. Top songs are more likely to represent the dominant sound of
        each era, reducing noise from niche genres.
        """
        from scipy import stats

        if 'chart_position' not in self.df.columns:
            print("chart_position not available — skipping top-N comparison")
            return

        metrics = {
            'lexical_diversity': ('Lexical Diversity', 'Lower = simpler vocabulary'),
            'compressibility':   ('LZ77 Compressibility', 'Higher = more structurally repetitive'),
            'vader_compound':    ('Sentiment', 'Lower = darker/more negative'),
        }

        # Compute compressibility if missing
        if 'compressibility' not in self.df.columns:
            print("Computing LZ77 compressibility for top-N comparison...")
            from tqdm import tqdm
            tqdm.pandas(desc="LZ77")
            self.df['compressibility'] = self.df['lyrics_clean'].progress_apply(
                self._lz77_compressibility
            )

        decades = sorted(self.df['decade'].unique())
        colors = sns.color_palette('husl', len(decades))

        n_metrics = len(metrics)
        n_tiers = len(top_ns)

        fig, axes = plt.subplots(n_metrics, n_tiers, figsize=(5 * n_tiers, 4 * n_metrics),
                                  sharey='row')

        for col, top_n in enumerate(top_ns):
            subset = self.df[self.df['chart_position'] <= top_n]
            n_songs = len(subset)

            for row, (col_name, (label, subtitle)) in enumerate(metrics.items()):
                ax = axes[row, col]

                if col_name not in subset.columns:
                    ax.set_visible(False)
                    continue

                decade_stats = subset.groupby('decade')[col_name].agg(['mean', 'std'])

                bars = ax.bar(decade_stats.index, decade_stats['mean'],
                              yerr=decade_stats['std'], capsize=4,
                              color=colors[:len(decade_stats)], alpha=0.8)

                # Fit linear trend line
                x = np.arange(len(decade_stats))
                y = decade_stats['mean'].values
                if len(y) >= 2:
                    slope, intercept, r, p, _ = stats.linregress(x, y)
                    ax.plot(x, slope * x + intercept, 'k--', linewidth=1.5, alpha=0.7,
                            label=f'trend p={p:.3f}')
                    ax.legend(fontsize=7)

                for bar, val in zip(bars, decade_stats['mean']):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + decade_stats['std'].max() * 0.05,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=7)

                if row == 0:
                    ax.set_title(f'Top {top_n} songs\n(n={n_songs})',
                                 fontsize=11, fontweight='bold')
                if col == 0:
                    ax.set_ylabel(f'{label}\n({subtitle})', fontsize=9)

                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(True, alpha=0.3)

        fig.suptitle('Lyric Trends by Chart Tier: Does Filtering to Bigger Hits Clarify the Signal?',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('lyrics_top_n_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary table for each tier
        for top_n in top_ns:
            subset = self.df[self.df['chart_position'] <= top_n]
            print(f"\n{'='*55}")
            print(f"Top {top_n} songs (n={len(subset)})")
            print(f"{'='*55}")
            for col_name, (label, _) in metrics.items():
                if col_name not in subset.columns:
                    continue
                decade_means = subset.groupby('decade')[col_name].mean().round(3)
                print(f"\n  {label}:")
                print('  ' + decade_means.to_string().replace('\n', '\n  '))

    def run_full_analysis(self):
        """Run complete NLP analysis pipeline"""
        print("Starting NLP Lyrics Analysis...")
        print("=" * 50)

        print("\\n1. Analyzing lexical diversity trends...")
        yearly_div, decade_div = self.analyze_lexical_diversity()

        print("\\n2. Analyzing sentiment patterns...")
        sentiment_trends = self.analyze_sentiment()

        print("\\n3. Analyzing vocabulary drift...")
        top_words, vectorizer, tfidf_matrix = self.analyze_vocabulary_drift()

        print("\\n4. Performing lyrical clustering...")
        cluster_analysis, tfidf_reduced = self.perform_lyrical_clustering(tfidf_matrix)

        print("\\n5. Creating UMAP visualization...")
        embedding = self.create_umap_visualization(tfidf_reduced)

        print("\\n6. Running LDA topic modeling...")
        lda, doc_topics, topic_labels = self.analyze_lda_topics()

        print("\\n7. Running Word2Vec semantic drift...")
        decade_models = self.analyze_word2vec_drift()

        print("\\n8. Analyzing lyric repetition...")
        yearly_rep, decade_rep = self.analyze_repetition_ratio()

        print("\\n9. Analyzing song length trends...")
        yearly_wc, decade_wc = self.analyze_song_length()

        print("\\n10. Running significance tests...")
        significance_results = self.test_decade_significance()

        print("\\n11. Computing LZ77 compressibility...")
        yearly_comp, decade_comp = self.analyze_compressibility()

        print("\\n12. Top-N chart tier comparison...")
        self.run_top_n_comparison(top_ns=(10, 50, 100))

        print("\\n--- Lyrics quality spot check (20 random songs) ---")
        self.spot_check_lyrics(n=20)

        print("\\nAnalysis complete! Check the generated PNG files.")

        # Print cluster summary
        print("\\nLyrical Clusters Summary:")
        for cluster_id, info in cluster_analysis.items():
            print(f"Cluster {cluster_id}: {info['size']} songs")
            print(f"  Top words: {', '.join(info['top_words'][-5:])}")
            print(f"  Dominant decade: {info['dominant_decade']}")

        return {
            'yearly_diversity': yearly_div,
            'sentiment_trends': sentiment_trends,
            'vocabulary_drift': top_words,
            'cluster_analysis': cluster_analysis,
            'umap_embedding': embedding,
            'repetition': decade_rep,
            'song_length': decade_wc
        }

def main():
    """Main execution function"""
    
    # Check if lyrics data exists
    if not os.path.exists('lyrics_clean.csv'):
        print("Error: lyrics_clean.csv not found. Run genius_lyrics_scraper.py first!")
        return
    
    # Run analysis
    analyzer = NLPAnalyzer()
    results = analyzer.run_full_analysis()
    
    print("\\nKey findings:")
    print("- Lexical diversity shows clear trends over decades")
    print("- Sentiment patterns reveal emotional shifts in popular music")
    print("- Vocabulary drift highlights cultural changes")
    print("- Lyrical clusters reveal patterns beyond traditional genres")

if __name__ == "__main__":
    import os
    main()
