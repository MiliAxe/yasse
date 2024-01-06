import heapq
import math
from typing import Dict, List
from scipy.sparse import csr_array

import numpy as np
import numpy.typing as npt

from modules.dataprocessor import DataProcessor, tokenize


class SearchEngine:
    # tf_idf_dict: Dict[str, npt.NDArray]
    tf_idf_sentences_dict: Dict[str, npt.NDArray]

    def __init__(self, paths: List[str]) -> None:
        self.tf_idf_dict = dict()
        self.dp = DataProcessor()
        self.dp.paths = paths
        self.dp.generate()

    # there's still room for performance improvement
    def tf_word_in_all_docs(self, word):
        return self.dp.docs_occurances_list(word) / self.dp.word_count_in_each_doc

    @staticmethod
    def tf_word_in_one_sentence(word, sentence_str):
        words = tokenize(sentence_str)
        words_count = len(words)

        return words.count(word) / words_count

    def tf_word_in_doc_sentences(self, word, doc_index):
        word = word.lower()
        sentences = self.dp.document_sentences(doc_index)
        return np.array(
            [self.tf_word_in_one_sentence(word, sentence) for sentence in sentences],
            dtype=float,
        )

    def idf_word_in_all_docs(self, word): 
        return math.log(len(self.dp.paths) / (len(self.dp.occur_dict[word]) + 1))

    def idf_word_in_doc(self, word, doc_index):
        return math.log(
            len(self.dp.document_sentences(doc_index))
            / (self.dp.count_sentences_with_word_in_document(word, doc_index) + 1) 
        )

    def calculate_tf_idf_all_docs(self):
        # self.tf_idf_dict = {word: self.tf_word_in_all_docs(
        #     word) * self.idf_word_in_all_docs(word) for word in self.dp.occur_dict}

        self.tf_idf_dict = {
            word: csr_array(
                self.tf_word_in_all_docs(word) * self.idf_word_in_all_docs(word)
            )
            for word in self.dp.occur_dict
        }

    def calculate_tf_idf_doc(self, doc_index):
        self.tf_idf_sentences_dict = {
            word.lower(): self.tf_word_in_doc_sentences(word.lower(), doc_index)
            * self.idf_word_in_doc(word.lower(), doc_index)
            for word in self.dp.document_words(doc_index)
        }

    def get_tf_idf_words_of_doc(self, doc_index) -> Dict[str, csr_array]: 
        doc_words = self.dp.document_words(doc_index)
        return {word: self.tf_idf_dict[word][0, doc_index] for word in doc_words}

    def get_tf_idf_words_of_sentence_one_doc(self, doc_index, sentence_index):
        doc_words = self.dp.document_words(doc_index)
        return {word: self.tf_idf_sentences_dict[word][sentence_index] for word in doc_words}

    def get_tf_idf_words_of_sentence(self, sentence):
        # causes a lot of performance issues
        # closest_sentence = [self.dp.get_closest_word_all_docs(word) for word in tokenize(query)]
        closest_sentence = tokenize(sentence)
        return {word: self.tf_word_in_one_sentence(word, sentence) * self.idf_word_in_all_docs(word) for word in 
                        closest_sentence} 

    def calculate_cosine_similarity(self, tf_idf_query, tf_idf_corpus):
        # calculating the numerator

        common_words_doc_query = set(tf_idf_corpus) & set(tf_idf_query)
        numerator = 0
        for word in common_words_doc_query:
            numerator += tf_idf_query[word] * tf_idf_corpus[word]

        # calculating the denumerator
        sum1 = sum([tf_idf_query[x] ** 2 for x in list(tf_idf_query)])
        sum2 = sum([tf_idf_corpus[x] ** 2 for x in list(tf_idf_corpus)])
        denominator = math.sqrt(sum1) + math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def cosine_similarity_of_doc(self, query, doc_index): 
        tf_idf_query = self.get_tf_idf_words_of_sentence(query) 

        tf_idf_doc = self.get_tf_idf_words_of_doc(doc_index)  

        return self.calculate_cosine_similarity(tf_idf_query, tf_idf_doc) 

    def cosine_similarities_docs(self, query): 
        return [
            (index, self.dp.paths[index], 
            self.cosine_similarity_of_doc(query, index)) 
            for index in range(len(self.dp.paths)) 
        ]

    def cosine_similarity_of_sentence(self, query, doc_index, sentence_index):
        tf_idf_query = self.get_tf_idf_words_of_sentence(query)

        tf_idf_sentence = self.get_tf_idf_words_of_sentence_one_doc(doc_index, sentence_index)

        return self.calculate_cosine_similarity(tf_idf_query, tf_idf_sentence)

    def cosine_similarities_doc_sentences(self, query, doc_index):
        self.calculate_tf_idf_doc(doc_index)
        return [
            (index, self.cosine_similarity_of_sentence(query, doc_index, index))
            for index in range(len(self.dp.document_sentences(doc_index)))
        ]

    def get_word_with_highest_tf_in_doc(self, doc_index):
        tf_dict = {}
        for sentence in self.dp.sentences[doc_index]:
            for word in sentence:
                if word in tf_dict:
                    tf_dict[word] += 1
                else:
                    tf_dict[word] = 1

        highest_tf_word = max(tf_dict, key=tf_dict.get)
        return highest_tf_word

    def get_top_five_idf_words_in_doc(self, doc_index):
        idf_dict = {
            word: self.idf_word_in_all_docs(word)
            for word in self.dp.document_words(doc_index)
        }
        top_five_idf_words = heapq.nlargest(5, idf_dict, key=idf_dict.get)
        return top_five_idf_words
