"""
Lexically analyze the .text sections in the dataset for entries with veryGood: True and quality >= +1.
Buckets similar texts together using KMeans clustering and extracts the top terms per cluster
to label issues based on significant properties.
"""
import ast
from regex import regex as re
import json
import os
from collections import defaultdict
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import strip_markdown
from markdown_it import MarkdownIt
from mdit_plain.renderer import RendererPlain
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.footnote import footnote_plugin
#import re
# # Extract themes and similar interests from the text column
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(df['text'])
# # Perform KMeans clustering
# kmeans = KMeans(n_clusters=5, random_state=42)
# df['cluster'] = kmeans.fit_predict(X)
# # Summarize themes by cluster
# themes = df.groupby('cluster')['text'].apply(lambda texts: ' '.join(texts)).reset_index()
# themes.columns = ['Cluster', 'Themes']
# # Assign results back to df
# df = pd.DataFrame({'Answer': themes['Themes']})
from transformers import pipeline
from collections import Counter

def generate_insights(group_texts):
    words = ' '.join(group_texts).lower().split()
    word_freq = Counter(words)
    
    return {
        'most_common': word_freq.most_common(5),
        'lexical_diversity': len(word_freq)/len(words),
        'avg_sentence_length': sum(len(t.split()) for t in group_texts)/len(group_texts)
    }


NUM_GROUPS=256

DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/export/md0/dataset.jsonl")
import re

class RobustMarkdownToTextConverter:
    def __init__(self):
        # Compile all regex patterns once
        self.patterns = {
            'links': re.compile(
                r'\[([^\]]*)\]\(([^)\s]+(?:[^)]*))\)',
                re.VERBOSE
            ),
            'images': re.compile(r'!\[([^\]]*)\]\([^)]*\)'),
            'inline_code': re.compile(r'`([^`]+)`'),
            'code_blocks': re.compile(r'```.*?\n[\s\S]*?\n```', re.DOTALL),
            'headers': re.compile(r'#+\s*(.*?)\s*#*$', re.MULTILINE),
            'bold': re.compile(r'(\*\*|__)(.*?)\1'),
            'italic': re.compile(r'(\*|_)(.*?)\1'),
            'blockquotes': re.compile(r'^\s*>.*$', re.MULTILINE),
            'lists': re.compile(r'^\s*[-*+]\s+|\s*\d+\.\s+', re.MULTILINE),
            'whitespace': re.compile(r'\n\s*\n')
        }
    
    def convert(self, markdown_text):
        text = markdown_text
        
        # Processing order matters - do code blocks first
        text = self.patterns['code_blocks'].sub('', text)
        text = self.patterns['inline_code'].sub(r'\1', text)
        text = self.patterns['images'].sub(r'\1', text)
        text = self.patterns['links'].sub(r'\1', text)
        text = self.patterns['headers'].sub(r'\1', text)
        text = self.patterns['bold'].sub(r'\2', text)
        text = self.patterns['italic'].sub(r'\2', text)
        text = self.patterns['blockquotes'].sub('', text)
        text = self.patterns['lists'].sub('', text)
        text = self.patterns['whitespace'].sub('\n\n', text)
        
        return text.strip()
    

    import re

def strip_all_markdown(text):
    """
    Nuclear option - removes ALL Markdown formatting, URLs, and GitHub-specific tags
    while preserving the readable text content. Handles high-density link cases.
    """
    # Remove reference-style links [text][id] → text
    text = re.sub(r'\[([^\]]+)\]\[[^\]]+\]', r'\1', text)
    
    # Remove inline links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove auto-links <url> → ''
    text = re.sub(r'<https?://[^>]+>', '', text)
    
    # Remove images ![alt](url) → ''
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
    
    # Remove GitHub-specific tags
    text = re.sub(r'(<!--.*?-->)|(^|\W)@[a-zA-Z0-9-]+|#[0-9]+|\bGH-[0-9]+\b', '', text)
    
    # Remove bare URLs (not in Markdown format)
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove any remaining angle brackets or artifacts
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace artifacts
    text = ' '.join(text.split())
    
    return text.strip()

# Example with ultra-dense Markdown
remove_chars = {'[', '(', '\\', '}', '{', ')', ';', '\\', '`', ']', }  # Characters to remove
translator = str.maketrans('', '', ''.join(remove_chars))

def clean_text(text):
    return text.translate(translator)


def load_filtered_entries(dataset_path: str):
    """
    Load and filter dataset entries where:
      - "veryGood" is True
      - "quality" is >= +1
    
    Returns:
        List of tuples (entry_id, text)
    """
    entries = []
    with open(dataset_path, 'r') as reader:
        for line_idx, xxxx in enumerate(reader):
            obj = json.loads(xxxx)
            # Validate the criteria
            if obj["metadata"]["veryGood"] and obj["metadata"]["quality"] > 0:
                # Create a tuple with (entry_id, text)
                entry_id = line_idx  # Or use a more meaningful ID if available
                text = clean_text(strip_all_markdown(obj["text"]))
                entries.append((entry_id, text))
    return entries

def perform_text_clustering(texts, num_clusters=NUM_GROUPS):
    """
    Vectorize texts using TF-IDF and cluster them with KMeans.
    
    Args:
        texts: List of text documents.
        num_clusters: Number of clusters to use.
    
    Returns:
        kmeans: fitted KMeans model
        vectorizer: fitted TFIDF vectorizer
        X: TF-IDF feature matrix
    """
    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", max_df=0.8, min_df=2)
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    return kmeans, vectorizer, X

def get_top_terms_per_cluster(kmeans, vectorizer, num_terms=NUM_GROUPS):
    """
    For each cluster, extract the top TF-IDF terms.
    
    Returns:
        Dictionary mapping cluster label to list of top terms.
    """
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    cluster_terms = {}
    for i in range(kmeans.n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :num_terms]]
        cluster_terms[i] = top_terms
    return cluster_terms

def bucket_entries_by_cluster(kmeans, texts, entry_ids):
    """
    Bucket the entries into clusters.
    
    Returns:
        Dictionary mapping cluster label to list of (entry_id, text)
    """
    clusters = defaultdict(list)
    labels = kmeans.labels_
    for entry_id, text, label in zip(entry_ids, texts, labels):
        clusters[label].append((entry_id, text))
    return clusters

def print_cluster_table(clusters, cluster_terms):
    """
    Print a table of clusters, grouping similar entries, with top terms as labels.
    """
    print("Cluster | Top Terms                     | Number of Entries")
    print("--------+-------------------------------+------------------")
    for label in sorted(clusters.keys()):
        top = ", ".join(cluster_terms.get(label, []))
        count = len(clusters[label])
        print(f"{label:^7} | {top:<29} | {count:^16}")
        for entry_id, text in clusters[label]:
            # Optionally, print a summary of each entry (first 80 characters)
            summary = text[:240].replace("\n", " ") + ("..." if len(text)>240 else "")
            print(f"        | [{entry_id}] {summary}")
        print("--------+-------------------------------+------------------")

def main():
    # Load and filter dataset
    entries = load_filtered_entries(DATASET_PATH)
    if not entries:
        print("No valid entries found matching criteria.")
        return
    
    # Split entries into ids and texts
    entry_ids, texts = zip(*entries) if entries else ([], [])
    
    # Perform clustering
    num_clusters = min(NUM_GROUPS, len(texts))  # Ensure we don't request more clusters than samples
    kmeans, vectorizer, X = perform_text_clustering(texts, num_clusters=num_clusters)
    
    # Get top terms per cluster for labeling issues
    cluster_terms = get_top_terms_per_cluster(kmeans, vectorizer, num_terms=256)
    
    # Bucket entries by cluster - pass all three required arguments
    clusters = bucket_entries_by_cluster(kmeans, texts, entry_ids)
    
    # Print cluster table: cluster label, top terms, count and entry summaries
    print_cluster_table(clusters, cluster_terms)

if __name__ == "__main__":
    main()
