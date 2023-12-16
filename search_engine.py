import math
from typing import Dict, List
import numpy.typing as npt
import numpy as np

from data_processor import DataProcessor

class SearchEngine:
    tf_idf_dict:Dict[str, npt.NDArray]

    def __init__(self, paths:List[str]) -> None:
        self.tf_idf_dict = dict()
        self.dp = DataProcessor()
        for path in paths:
            self.dp.add_file(path)
        self.dp.generate()

    # theres still room for performance improvement
    def tf_word(self, word):
        return self.dp.docs_occurances_list(word)/self.dp.doc_wordcount_list

    def idf_word(self, word):
        return math.log(len(self.dp.paths)/(len(self.dp.occur_dict[word])+1))

    def calculate_tf_idf(self):
        self.tf_idf_dict = {word: self.tf_word(word)*self.idf_word(word) for word in self.dp.occur_dict}
        print(sorted(self.tf_idf_dict.items(), key=lambda item: np.sum(item[1]), reverse=True))
