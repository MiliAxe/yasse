from searchengine import SearchEngine
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy.sparse import csr_array, csr_matrix, vstack
from sys import argv
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from argparser import parser

class Visualizer:
    tf_idf_matrix: csr_matrix
    tf_idf_word_index: Dict[int, str]
    # tf_idf_decomposed_matrix: csr_matrix
    searchengine: SearchEngine

    def __init__(self, paths: List[str]) -> None:
        self.searchengine = SearchEngine(paths)
        self.searchengine.calculate_tf_idf_all_docs()

    def generate_tfidf_word_dict(self):
        self.tf_idf_word_index = {index: word for index, word in enumerate(self.searchengine.tf_idf_dict)}

    def generate_tfidf_matrix(self):
        self.generate_tfidf_word_dict()
        tf_idf_arrays = []

        for index in self.tf_idf_word_index:
            current_word = self.tf_idf_word_index[index]
            tf_idf_arrays.append(self.searchengine.tf_idf_dict[current_word])

        stacked_arrays = vstack(tf_idf_arrays)
        self.tf_idf_matrix = csr_matrix(stacked_arrays.transpose())

    def decompose_to_2d(self):
        self.generate_tfidf_matrix()
        decomposer = TruncatedSVD(n_components=2)
        self.tf_idf_decomposed_matrix = decomposer.fit_transform(self.tf_idf_matrix)

    def compute_clusters(self, cluster_count):
        self.decompose_to_2d()
        
        kmeans = KMeans(n_clusters=cluster_count, n_init=3, random_state=0)

        kmeans.fit(self.tf_idf_decomposed_matrix)

        self.cluster_centroids = kmeans.cluster_centers_
        self.cluster_labels = kmeans.labels_

    def plot_clusters(self):
        plt.scatter(self.tf_idf_decomposed_matrix[:, 0], self.tf_idf_decomposed_matrix[:, 1], c=self.cluster_labels, cmap="viridis", edgecolor="k")
        plt.show()
        
if __name__ == "__main__":
    args = parser.parse_args()
    visualizer = Visualizer(args.files)

    visualizer.compute_clusters(3)
    visualizer.plot_clusters()
