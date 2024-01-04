from searchengine import SearchEngine
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy.sparse import csr_array, csr_matrix, vstack
from sys import argv
from sklearn.decomposition import TruncatedSVD

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
        
        # 2d_decomposer = TruncatedSVD(n_components=2)
        
        self.tf_idf_decomposed_matrix = TruncatedSVD(n_components=2).fit_transform(self.tf_idf_matrix)

    def computer_clusters(self):
        self.decompose_to_2d()

        






if __name__ == "__main__":
    paths = [f"data/document_{index}.txt" for index in range(1000)]
    visualizer = Visualizer(paths)
    visualizer.generate_tfidf_matrix()
    new_matrix = TruncatedSVD(n_components=2).fit_transform(visualizer.tf_idf_matrix)
    plt.style.use("_mpl-gallery")
    _, ax = plt.subplots()
    scatter = ax.scatter(new_matrix[:, 0], new_matrix[:, 1])
    plt.show()
    # print(visualizer.tf_idf_word_index)
