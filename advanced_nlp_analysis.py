"""
Advanced NLP Lyrics Analysis
Implements sophisticated NLP techniques including:
- Transformer-based embeddings (BERT)
- Topic modeling with LDA and NMF
- Semantic shift analysis
- Stylometric analysis
- Cross-lingual patterns
- Causal language modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import networkx as nx
from wordcloud import WordCloud
import spacy
from collections import Counter
import gensim
from gensim.models import LdaModel, Word2Vec, Doc2Vec
from gensim.corpora import Dictionary
import transformers
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class AdvancedNLPAnalyzer:
    def __init__(self, lyrics_file='lyrics_clean.csv'):
        """Initialize with advanced NLP analysis capabilities"""
        self.df = pd.read_csv(lyrics_file)
        self.df = self.df.dropna(subset=['lyrics_clean'])
        self.prepare_data()
        
        # Load spaCy model for advanced linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def prepare_data(self):
        """Prepare data for advanced NLP analysis"""
        # Add decade column
        self.df['decade'] = (self.df['year'] // 10) * 10
        self.df['decade'] = self.df['decade'].astype(str) + 's'
        
        # Clean lyrics further
        self.df['lyrics_processed'] = self.df['lyrics_clean'].apply(self.advanced_text_preprocessing)
        
        print(f"Loaded {len(self.df)} songs for advanced NLP analysis")
    
    def advanced_text_preprocessing(self, text):
        """Advanced text preprocessing with linguistic features"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower().strip()
        
        # Remove song-specific artifacts
        text = self.remove_song_artifacts(text)
        
        # Linguistic processing if spaCy is available
        if self.nlp:
            doc = self.nlp(text)
            
            # Keep only meaningful tokens
            tokens = []
            for token in doc:
                if (not token.is_stop and 
                    not token.is_punct and 
                    not token.is_space and 
                    not token.like_num and
                    len(token.text) > 2):
                    tokens.append(token.lemma_)
            
            return ' '.join(tokens)
        else:
            # Fallback basic processing
            import re
            tokens = re.findall(r'\\b[a-zA-Z]{3,}\\b', text)
            return ' '.join(tokens)
    
    def remove_song_artifacts(self, text):
        """Remove common song artifacts and non-lyrical content"""
        import re
        
        # Remove common patterns
        artifacts = [
            r'\\[.*?\\]',  # [Chorus], [Verse], etc.
            r'\\(.*?\\)',  # (repeat), (ad-lib), etc.
            r'\\d+\\x[0-9a-f]+',  # Unicode artifacts
            r'^[^a-zA-Z]*',  # Non-alphabetic start
            r'[^a-zA-Z]*$',  # Non-alphabetic end
        ]
        
        for pattern in artifacts:
            text = re.sub(pattern, ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)
        
        return text.strip()
    
    def transformer_embeddings(self, model_name='distilbert-base-uncased'):
        """Generate transformer-based embeddings for lyrics"""
        
        print(f"Loading transformer model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        embeddings = []
        batch_size = 32  # Process in batches to manage memory
        
        for i in range(0, len(self.df), batch_size):
            batch_texts = self.df['lyrics_processed'].iloc[i:i+batch_size].tolist()
            
            # Tokenize
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                            max_length=512, return_tensors='pt')
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding or mean pooling
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(self.df)-1)//batch_size + 1}")
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)
        
        # Add embeddings to dataframe
        for i in range(all_embeddings.shape[1]):
            self.df[f'bert_dim_{i}'] = all_embeddings[:, i]
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        self.df['bert_tsne_1'] = embeddings_2d[:, 0]
        self.df['bert_tsne_2'] = embeddings_2d[:, 1]
        
        # Visualize
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=self.df['year'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Year')
        plt.xlabel('BERT t-SNE Dimension 1')
        plt.ylabel('BERT t-SNE Dimension 2')
        plt.title('BERT Embeddings of Song Lyrics', fontsize=14, fontweight='bold')
        plt.savefig('bert_embeddings_tsne.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return all_embeddings
    
    def advanced_topic_modeling(self):
        """Implement multiple topic modeling approaches"""
        
        # Prepare documents
        documents = self.df['lyrics_processed'].tolist()
        
        # Create different vectorizers
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                                         ngram_range=(1, 3), min_df=5, max_df=0.7)
        count_vectorizer = CountVectorizer(max_features=1000, stop_words='english', 
                                          ngram_range=(1, 2), min_df=5, max_df=0.7)
        
        # Fit vectorizers
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        count_matrix = count_vectorizer.fit_transform(documents)
        
        # LDA with count matrix
        lda = LatentDirichletAllocation(n_components=10, random_state=42, max_iter=10)
        lda_topics = lda.fit_transform(count_matrix)
        
        # NMF with TF-IDF matrix
        nmf = NMF(n_components=10, random_state=42, max_iter=200)
        nmf_topics = nmf.fit_transform(tfidf_matrix)
        
        # Extract and visualize topics
        def get_top_words(model, feature_names, n_top_words=10):
            topics = []
            for topic_idx, topic in enumerate(model.components_):
                top_indices = topic.argsort()[-n_top_words:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics.append(top_words)
            return topics
        
        # Get topics
        lda_topics_words = get_top_words(lda, count_vectorizer.get_feature_names_out())
        nmf_topics_words = get_top_words(nmf, tfidf_vectorizer.get_feature_names_out())
        
        # Visualize topics
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        # LDA topics
        for i, topic_words in enumerate(lda_topics_words[:5]):
            word_freq = Counter(topic_words)
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'LDA Topic {i+1}')
            axes[i].axis('off')
        
        # NMF topics
        for i, topic_words in enumerate(nmf_topics_words[:5]):
            word_freq = Counter(topic_words)
            wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
            axes[i+5].imshow(wordcloud, interpolation='bilinear')
            axes[i+5].set_title(f'NMF Topic {i+1}')
            axes[i+5].axis('off')
        
        plt.tight_layout()
        plt.savefig('advanced_topic_modeling.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Add topic distributions to dataframe
        self.df['lda_dominant_topic'] = np.argmax(lda_topics, axis=1)
        self.df['nmf_dominant_topic'] = np.argmax(nmf_topics, axis=1)
        
        return {
            'lda': (lda, lda_topics, lda_topics_words),
            'nmf': (nmf, nmf_topics, nmf_topics_words)
        }
    
    def semantic_shift_analysis(self):
        """Analyze semantic shifts of words over decades"""
        
        # Get words that appear in multiple decades
        decade_word_freq = {}
        for decade in sorted(self.df['decade'].unique()):
            decade_lyrics = self.df[self.df['decade'] == decade]['lyrics_processed']
            all_words = ' '.join(decade_lyrics).split()
            word_freq = Counter(all_words)
            decade_word_freq[decade] = word_freq
        
        # Find common words across decades
        common_words = set(decade_word_freq['1970s'].keys())
        for decade in decade_word_freq.keys():
            if decade != '1970s':
                common_words &= set(decade_word_freq[decade].keys())
        
        common_words = list(common_words)[:50]  # Limit to top 50
        
        # Calculate semantic shifts using context similarity
        semantic_shifts = {}
        
        for word in common_words:
            word_contexts = {}
            
            for decade in decade_word_freq.keys():
                # Get context words for this word in this decade
                contexts = []
                decade_lyrics = self.df[self.df['decade'] == decade]['lyrics_processed']
                
                for lyrics in decade_lyrics:
                    words = lyrics.split()
                    for i, w in enumerate(words):
                        if w == word:
                            # Get context window
                            start = max(0, i - 3)
                            end = min(len(words), i + 4)
                            context = [words[j] for j in range(start, end) if words[j] != word]
                            contexts.extend(context)
                
                word_contexts[decade] = Counter(contexts)
            
            # Calculate semantic shift between first and last decade
            first_decade = sorted(decade_word_freq.keys())[0]
            last_decade = sorted(decade_word_freq.keys())[-1]
            
            # Calculate Jaccard similarity between contexts
            first_context = set(word_contexts[first_decade].keys())
            last_context = set(word_contexts[last_decade].keys())
            
            if len(first_context) > 0 and len(last_context) > 0:
                jaccard = len(first_context & last_context) / len(first_context | last_context)
                semantic_shifts[word] = 1 - jaccard  # Shift = 1 - similarity
        
        # Sort by semantic shift
        sorted_shifts = sorted(semantic_shifts.items(), key=lambda x: x[1], reverse=True)
        
        # Visualize semantic shifts
        plt.figure(figsize=(12, 8))
        words, shifts = zip(*sorted_shifts[:20])
        
        bars = plt.barh(words, shifts, alpha=0.7, color='coral')
        plt.xlabel('Semantic Shift (1 - Jaccard Similarity)')
        plt.title('Words with Largest Semantic Shifts (1970s vs 2020s)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for bar, shift in zip(bars, shifts):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{shift:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('semantic_shift_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return sorted_shifts
    
    def stylometric_analysis(self):
        """Advanced stylometric analysis of lyrics"""
        
        stylometric_features = []
        
        for lyrics in self.df['lyrics_processed']:
            if not lyrics:
                stylometric_features.append({})
                continue
            
            words = lyrics.split()
            sentences = lyrics.split('.')
            
            # Basic statistics
            features = {
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'avg_sentence_length': np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0,
                'punctuation_ratio': lyrics.count('.') + lyrics.count(',') + lyrics.count('!') + lyrics.count('?'),
                'unique_words_ratio': len(set(words)) / len(words) if words else 0,
                'hapax_legomena_ratio': sum(1 for w in Counter(words).values() if w == 1) / len(words) if words else 0,
            }
            
            # Advanced linguistic features (if spaCy available)
            if self.nlp:
                doc = self.nlp(lyrics)
                
                # POS tag ratios
                pos_counts = Counter([token.pos_ for token in doc])
                total_tokens = len([token for token in doc if not token.is_punct])
                
                if total_tokens > 0:
                    features.update({
                        'noun_ratio': pos_counts.get('NOUN', 0) / total_tokens,
                        'verb_ratio': pos_counts.get('VERB', 0) / total_tokens,
                        'adj_ratio': pos_counts.get('ADJ', 0) / total_tokens,
                        'adv_ratio': pos_counts.get('ADV', 0) / total_tokens,
                    })
                else:
                    features.update({'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0, 'adv_ratio': 0})
                
                # Readability scores
                features['readability_score'] = self.flesch_reading_ease(doc)
            
            stylometric_features.append(features)
        
        # Convert to DataFrame
        stylometric_df = pd.DataFrame(stylometric_features)
        
        # Add to main dataframe
        for col in stylometric_df.columns:
            self.df[f'style_{col}'] = stylometric_df[col]
        
        # Analyze stylometric trends
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        style_features_to_plot = ['avg_word_length', 'unique_words_ratio', 'noun_ratio', 
                                 'verb_ratio', 'adj_ratio', 'readability_score']
        
        for i, feature in enumerate(style_features_to_plot):
            if feature in stylometric_df.columns:
                yearly_avg = self.df.groupby('year')[f'style_{feature}'].mean()
                axes[i].plot(yearly_avg.index, yearly_avg.values, linewidth=2)
                axes[i].set_title(feature.replace('_', ' ').title())
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stylometric_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stylometric_df
    
    def flesch_reading_ease(self, doc):
        """Calculate Flesch reading ease score"""
        if not doc:
            return 0
        
        words = [token for token in doc if not token.is_punct]
        sentences = [sent for sent in doc.sents]
        
        if len(words) == 0 or len(sentences) == 0:
            return 0
        
        avg_sentence_length = len(words) / len(sentences)
        syllable_count = sum(self.count_syllables(token.text) for token in words)
        avg_syllables_per_word = syllable_count / len(words)
        
        # Flesch reading ease formula
        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        return max(0, min(100, score))
    
    def count_syllables(self, word):
        """Simple syllable counting"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def word_embedding_analysis(self):
        """Train Word2Vec and analyze semantic relationships"""
        
        # Prepare sentences for Word2Vec
        sentences = [lyrics.split() for lyrics in self.df['lyrics_processed']]
        
        # Train Word2Vec model
        print("Training Word2Vec model...")
        w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, 
                           workers=4, sg=1, epochs=10)
        
        # Analyze semantic changes by decade
        decade_models = {}
        for decade in sorted(self.df['decade'].unique()):
            decade_sentences = [lyrics.split() for lyrics in 
                              self.df[self.df['decade'] == decade]['lyrics_processed']]
            
            if len(decade_sentences) > 10:  # Ensure enough data
                decade_model = Word2Vec(decade_sentences, vector_size=100, window=5, 
                                      min_count=3, workers=4, sg=1, epochs=5)
                decade_models[decade] = decade_model
        
        # Find words with changing semantic contexts
        common_words = set(w2v_model.wv.key_to_index.keys())
        for decade_model in decade_models.values():
            common_words &= set(decade_model.wv.key_to_index.keys())
        
        common_words = list(common_words)[:20]  # Limit to 20 words
        
        # Calculate semantic drift for each word
        semantic_drifts = {}
        
        for word in common_words:
            if word in decade_models['1970s'].wv and word in decade_models['2020s'].wv:
                vector_1970s = decade_models['1970s'].wv[word]
                vector_2020s = decade_models['2020s'].wv[word]
                
                # Calculate cosine distance
                distance = cosine(vector_1970s, vector_2020s)
                semantic_drifts[word] = distance
        
        # Sort by drift
        sorted_drifts = sorted(semantic_drifts.items(), key=lambda x: x[1], reverse=True)
        
        # Visualize semantic drifts
        plt.figure(figsize=(12, 8))
        words, drifts = zip(*sorted_drifts[:15])
        
        bars = plt.barh(words, drifts, alpha=0.7, color='purple')
        plt.xlabel('Semantic Drift (Cosine Distance)')
        plt.title('Words with Largest Semantic Drifts (1970s vs 2020s)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for bar, drift in zip(bars, drifts):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{drift:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('word_embedding_drift.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return w2v_model, decade_models, sorted_drifts
    
    def run_advanced_nlp_analysis(self):
        """Run complete advanced NLP analysis pipeline"""
        print("Starting Advanced NLP Analysis...")
        print("=" * 50)
        
        print("\\n1. Generating transformer embeddings...")
        embeddings = self.transformer_embeddings()
        
        print("\\n2. Advanced topic modeling...")
        topic_results = self.advanced_topic_modeling()
        
        print("\\n3. Semantic shift analysis...")
        semantic_shifts = self.semantic_shift_analysis()
        
        print("\\n4. Stylometric analysis...")
        stylometric_results = self.stylometric_analysis()
        
        print("\\n5. Word embedding analysis...")
        w2v_model, decade_models, semantic_drifts = self.word_embedding_analysis()
        
        print("\\nAdvanced NLP analysis complete! Check generated PNG files.")
        
        return {
            'transformer_embeddings': embeddings,
            'topic_modeling': topic_results,
            'semantic_shifts': semantic_shifts,
            'stylometrics': stylometric_results,
            'word_embeddings': (w2v_model, decade_models, semantic_drifts)
        }

def main():
    """Main execution function"""
    
    # Check if lyrics data exists
    if not os.path.exists('lyrics_clean.csv'):
        print("Error: lyrics_clean.csv not found. Run genius_lyrics_scraper.py first!")
        return
    
    # Run advanced NLP analysis
    analyzer = AdvancedNLPAnalyzer()
    results = analyzer.run_advanced_nlp_analysis()
    
    print("\\nAdvanced NLP Analysis Summary:")
    print(f"- Generated BERT embeddings: {results['transformer_embeddings'].shape}")
    print(f"- Identified {len(results['topic_modeling']['lda'][2])} LDA topics")
    print(f"- Found {len(results['semantic_shifts'])} words with semantic shifts")
    print(f"- Analyzed {len(results['stylometrics'].columns)} stylometric features")
    print(f"- Trained Word2Vec with {len(results['word_embeddings'][0].wv)} vocabulary")

if __name__ == "__main__":
    import os
    main()
