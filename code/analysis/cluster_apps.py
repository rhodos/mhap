import pandas as pd
import numpy as np
import os
import ast
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# Configuration
NUMBER_OF_CLUSTERS = 50

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = script_dir + "/../../data/step3_llm_annotations/"
output_dir = script_dir + "/../../data/step4_clustering_results/"

INPUT_FILE = input_dir + 'structured_data.tsv'
OUTPUT_CLUSTER_FILE = output_dir + f'apps_clustered_{NUMBER_OF_CLUSTERS}_{timestamp}.tsv'
OUTPUT_CLUSTER_INFO_FILE = output_dir + f'cluster_info_{NUMBER_OF_CLUSTERS}_{timestamp}.tsv'


def safe_literal_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return [] # Return empty list for invalid strings
    return x # Return as is if not a string

def count_list_items(series: pd.Series) -> dict:
    """
    Counts the occurrences of individual items within lists in a pandas Series.

    Args:
        series: A pandas Series where each element is a list of items.

    Returns:
        A dictionary with items as keys and their counts as values.
    """
    item_counts = {}
    for item_list in series:
        if isinstance(item_list, list):
            for item in item_list:
                item_counts[item] = item_counts.get(item, 0) + 1
        elif isinstance(item_list, str):
            print('str')
            item_list = [item_list]
            for item in item_list:
                item_counts[item] = item_counts.get(item, 0) + 1
    return item_counts

def get_top_items(item_counts: dict, n: int = 5) -> list:
    """
    Gets the top N items from a dictionary based on their counts.

    Args:
        item_counts: A dictionary with items as keys and their counts as values.
        n: The number of top items to retrieve.

    Returns:
        A list of tuples, where each tuple contains an item and its count,
        sorted in descending order of counts.
    """
    sorted_items = sorted(item_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_items[:n]

if __name__ == "__main__":

    # Load the data
    df = pd.read_csv(INPUT_FILE, sep='\t')

    # Combine relevant text columns
    df['combined_text'] = df['title'].fillna('') + ' ' + \
                        df['description'].fillna('') + ' ' + \
                        df['features'].fillna('') + ' ' + \
                        df['target_demographic'].fillna('') + ' ' + \
                        df['indication'].fillna('')

    # Text Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # """# Explore optimal cluster number via Silhouette Scores"""

    # range_n_clusters = range(5, 100, 5)
    # silhouette_scores = []

    # for n_clusters in range_n_clusters:
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    #     cluster_labels = kmeans.fit_predict(tfidf_matrix)
    #     score = silhouette_score(tfidf_matrix, cluster_labels)
    #     silhouette_scores.append(score)

    # plt.figure(figsize=(10, 6))
    # plt.plot(range_n_clusters, silhouette_scores)
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("Average Silhouette Score")
    # plt.title("Silhouette Scores vs. Number of Clusters")
    # plt.show()

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=NUMBER_OF_CLUSTERS, random_state=0, n_init=10)
    kmeans.fit(tfidf_matrix)
    df['Cluster'] = kmeans.labels_

    # Ensure list columns are properly formatted
    df['features'] = df['features'].apply(safe_literal_eval)
    df['indication'] = df['indication'].apply(safe_literal_eval)
    df['target_demographic'] = df['target_demographic'].apply(safe_literal_eval)

    # Calculate silhouette coefficients for each sample and add to dataframe
    df['Cluster_Silhouette_Score'] = silhouette_samples(tfidf_matrix, df['Cluster'].astype(int))

    # Calculate the average silhouette score for each cluster
    cluster_silhouette_scores = df.groupby('Cluster')['Cluster_Silhouette_Score'].mean()

    # Sort by silhouette score
    sorted_clusters = cluster_silhouette_scores.sort_values(ascending=False).index

    cluster_info = pd.DataFrame(columns=['cluster', 'n_apps', 'cumulative_count', 'silhouette_score', 'app_names', 'top_terms', 'top_features', 'top_indications', 'top_demographics'])

    # Analyze clusters
    cumulative_app_count = 0
    #for i in range(number_of_clusters):
    for i in sorted_clusters:
        cluster_df = df[df['Cluster'] == i]

        if not cluster_df.empty:

            # Number of apps in cluster:
            n_apps = cluster_df.shape[0]
            cumulative_app_count += n_apps

            # Cluster silhouette score
            cluster_silhouette_score = cluster_silhouette_scores[i]

            # App names
            app_names = ", ".join(cluster_df.title)

            # Top terms in the cluster
            cluster_tfidf_matrix = tfidf_matrix[cluster_df.index]
            avg_tfidf_scores = np.array(cluster_tfidf_matrix.mean(axis=0)).flatten()
            top_terms_indices = avg_tfidf_scores.argsort()[::-1][:10]
            top_terms = [feature_names[idx] for idx in top_terms_indices.tolist()]
            top_scores = np.sort(avg_tfidf_scores)[::-1][:10]
            term_scores_dict = {term: score for term, score in zip(top_terms, top_scores)}

            # Top feature, indication and demographic counts
            features_counts = count_list_items(cluster_df['features'])
            indication_counts = count_list_items(cluster_df['indication'])
            demographic_counts = count_list_items(cluster_df['target_demographic'])

            cluster_num = f'cluster_{i}'
            cluster_info.loc[i] = [cluster_num, n_apps, cumulative_app_count,cluster_silhouette_score, app_names, term_scores_dict, features_counts, indication_counts, demographic_counts]

        else:
            print("Cluster is empty.")

    # Save the app clusters
    columns = ['id', 'title', 'Cluster', 'Cluster_Silhouette_Score']
    df[columns].to_csv(OUTPUT_CLUSTER_FILE, index=False, sep='\t', header=True)

    # Save the cluster_info (for annotation purposes)
    cluster_info.to_csv(OUTPUT_CLUSTER_INFO_FILE, index=False, sep='\t')    