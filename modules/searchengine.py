import math
from typing import Dict, List
import numpy.typing as npt
import numpy as np

from .dataprocessor import DataProcessor, SentencePosition, tokenize

class SearchEngine:
    tf_idf_dict:Dict[str, npt.NDArray]

    def __init__(self, paths:List[str]) -> None:
        self.tf_idf_dict = dict()
        self.dp = DataProcessor()
        self.dp.paths = paths
        self.dp.generate()

    # theres still room for performance improvement
    def tf_word(self, word):
        return self.dp.docs_occurances_list(word)/self.dp.word_count_in_each_doc

    def idf_word(self, word):
        return math.log(len(self.dp.paths)/(len(self.dp.occur_dict[word])+1))

    def calculate_tf_idf(self):
        self.tf_idf_dict = {word: self.tf_word(word)*self.idf_word(word) for word in self.dp.occur_dict}

    def sentence_tf_idf(self, sentence):
        sentence_tokens = set(tokenize(sentence))
        for d_index, sentences_list in enumerate(self.dp.sentences):
            for s_index, dp_sentence in enumerate(sentences_list):
                intersection = dp_sentence & sentence_tokens
                intersection_tf_idf = [self.tf_idf_dict[word] for word in intersection]

        # return [self.tf_idf_dict[word][doc_index] for word in sentence_tokens]

    def calculate_tf_idf_sentences(self):
        return [[self.tf_idf_dict[word][doc_index] for word in sentence] for doc_index in range(len(self.dp.paths)) for sentence in self.dp.sentences[doc_index]]
