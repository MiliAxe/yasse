from dataclasses import dataclass
from collections import defaultdict
import os, re, string
import numpy as np
import numpy.typing as npt
from typing import Dict, List
from difflib import get_close_matches
from scipy.sparse import lil_array, csr_array

from collections import Counter


def sentencize(str) -> List[str]:
    # return list(filter(None, re.split(r"\.\s+", str)))
    # return str.split('. ')
    return str.lower().split("\n")


def tokenize(str) -> List[str]:
    return str.lower().translate(str.maketrans("", "", string.punctuation)).split()
    # return str.lower().replace('. ', ' ').split()


@dataclass
class SentencePosition:
    doc_index: int
    sentence_index: int


class DataProcessor:
    paths: List[str]
    occur_dict: Dict[str, Dict[int, npt.NDArray]]
    sentences: list
    def __init__(self, path=None) -> None:
        self.occur_dict = defaultdict(dict)
        self.sentences_size = list()
        self.paths = list()
        self.sentences = list()
        if path is not None:
            self.add_file(path)

    def add_dir(self, dir) -> None:
        for file in os.listdir(dir):
            self.add_file(os.path.join(dir, file))

    def add_file(self, path) -> None:
        self.paths.append(path)

    def all_sentences(self):
        return [
            SentencePosition(doc_key, sentence_i)
            for word in self.occur_dict
            for doc_key, array in self.occur_dict[word].items()
            for sentence_i, occur in enumerate(array)
            if occur != 0
        ]

    def sentence_positions(self, word: str):
        return [
            SentencePosition(doc_key, sentence_i)
            for doc_key, array in self.occur_dict[word].items()
            for sentence_i, occur in enumerate(array)
            if occur != 0
        ]

    def generate(self):
        self.word_count_in_each_doc = np.zeros(len(self.paths), np.uint16)
        for doc_index, path in enumerate(self.paths):
            with open(path, encoding="utf-8") as file:
                data = file.read()
            sentences = sentencize(data)
            self.sentences_size.append(len(sentences))
            self.word_count_in_each_doc[doc_index] = len(tokenize(data))

            for token in set(tokenize(data)):
                # sentence count
                self.occur_dict[token][doc_index] = np.zeros(
                    self.sentences_size[doc_index], np.uint8
                )

            self.sentences.append([])
            for i, sentence in enumerate(sentences):
                sentence_tokens = tokenize(sentence)
                token_counts = Counter(sentence_tokens)
                self.sentences[-1].append(set(sentence_tokens))
                for token, count in token_counts.items():
                    self.occur_dict[token][doc_index][i] = count

    # def generate(self):
    #     self.word_count_in_each_doc = np.zeros(len(self.paths), np.uint16)
    #     for doc_index, path in enumerate(self.paths):
    #         with open(path, encoding='utf-8') as file:
    #             data = file.read()
    #         current_doc_sentences = sentencize(data)
    #         self.sentences.append(current_doc_sentences)
    #         self.sentences_size.append(len(current_doc_sentences))
    #         self.word_count_in_each_doc[doc_index] = len(tokenize(data))
    #
    #         for unique_word in set(tokenize(data)):
    #             if unique_word not in self.occur_dict:
    #                 # self.occur_dict[unique_word] = lil_array((len(self.paths), 80), dtype=np.uint8)
    #                 self.occur_dict[unique_word] = np.zeros((len(self.paths), 80), dtype=np.uint8)
    #
    #             for sentence_index, sentence in enumerate(current_doc_sentences):
    #                 sentence_tokens = tokenize(sentence)
    #                 self.occur_dict[unique_word][doc_index, sentence_index] = sentence_tokens.count(unique_word)
    #     for word in self.occur_dict:
    #         self.occur_dict[word] = csr_array(self.occur_dict[word])

    # development helpers
    def get_closest_word_all_docs(self, word):
        words = self.occur_dict.keys()
        return get_close_matches(word, words)[0]

    def get_closest_word_doc(self, word, doc_index):
        words = self.document_words(doc_index)
        return get_close_matches(word, words)[0]

    def check_word(self, word: str):
        if word not in self.occur_dict:
            raise KeyError(
                f'Error: word "{word}" not found in this instance of DataProcessor.'
            )

    def sentence_at(self, sp: SentencePosition):
        # with open(self.paths[sp.doc_index]) as file:
        #     data = file.read()
        return self.document_sentences(sp.doc_index)[sp.sentence_index]

    def document_sentences(self, doc_index):
        with open(self.paths[doc_index], encoding="utf-8") as file:
            data = file.read()
        return sentencize(data)

    def occurences(self, word: str) -> int:
        self.check_word(word)
        return np.sum(np.concatenate(list(self.occur_dict[word].values())))

    def document_occurences(self, word: str, index: int) -> int:
        if index >= len(self.paths) or index < 0:
            raise IndexError(
                f"Error: index is not valid. valid indexes for this instance are between 0 and {len(self.paths)-1}."
            )
        self.check_word(word)
        try:
            return np.sum(self.occur_dict[word][index])
        except KeyError:
            return 0

    def count_sentences_in_document(self, doc_index):
        return self.document_sentences(doc_index).count()

    def count_sentences_with_word_in_document(self, word, doc_index):
        try:   
            return len(self.occur_dict[word][doc_index].nonzero()[0])
        except KeyError:
            return 0

    def docs_occurances_list(self, word: str):
        self.check_word(word)
        output_arr = np.zeros(len(self.paths), np.uint16)
        for key, value in self.occur_dict[word].items():
            output_arr[key] = np.sum(value)
        return output_arr

    def document_words(self, doc_index):
        return [word for word in self.occur_dict if doc_index in self.occur_dict[word]]

    def __str__(self) -> str:
        output = ""
        for key, value in self.occur_dict.items():
            output += f"'{key}': {value}\n"
        return output
