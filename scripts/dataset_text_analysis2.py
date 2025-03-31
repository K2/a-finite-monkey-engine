"""
Lexically analyze the .text sections in the dataset for entries with veryGood: True and quality >= +1.
Buckets similar texts together using KMeans clustering and extracts the top terms per cluster
to label issues based on significant properties.
"""

import json
import jsonlines
import os
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../reports/dataset.jsonl")

def load_filtered_entries(dataset_path: str):
    """
    Load and filter dataset entries where:
      - "veryGood" is True
      - "quality" is >= +1
    
    Returns:
        List of tuples (entry_id, text)
    """
    entries = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader:
            # Validate the criteria
            if obj.get("veryGood") is True and obj.get("quality", 0) >= 1:
                text = obj.get("text", "").strip()
                if text:
                    entry_id = obj.get("id", "N/A")
                    entries.append((entry_id, text))
    return entries

def perform_text_clustering(texts, num_clusters=5):
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
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, min_df=2)
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans, vectorizer, X

def get_top_terms_per_cluster(kmeans, vectorizer, num_terms=5):
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
            summary = text[:80].replace("\n", " ") + ("..." if len(text)>80 else "")
            print(f"        | [{entry_id}] {summary}")
        print("--------+-------------------------------+------------------")

def main():
    # Load and filter dataset
    entries = load_filtered_entries(DATASET_PATH)
    if not entries:
        print("No valid entries found matching criteria.")
        return
    entry_ids, texts = zip(*entries)
    
    # Perform clustering
    num_clusters = 5  # You might adjust this based on data size or use elbow method
    kmeans, vectorizer, X = perform_text_clustering(texts, num_clusters=num_clusters)
    
    # Get top terms per cluster for labeling issues
    cluster_terms = get_top_terms_per_cluster(kmeans, vectorizer, num_terms=5)
    
    # Bucket entries by cluster
    clusters = bucket_entries_by_cluster(kmeans, texts, entry_ids)
    
    # Print cluster table: cluster label, top terms, count and entry summaries
    print_cluster_table(clusters, cluster_terms)

if __name__ == "__main__":
    main()
