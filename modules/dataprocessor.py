"""
This module provides a data processing class and related functions to process and analyze string data from text files. It includes capabilities for tokenizing text, locating sentences, counting word occurrences, and managing text data across multiple documents.

Functions:
    - sentencize(str) -> List[str]:
        Splits a string into sentences based on full stops followed by whitespace.

    - tokenize(str) -> List[str]:
        Converts a string to lowercase, removes punctuation, and splits it into a list of words.

Data Classes:
    - SentencePosition:
        A data class that stores a document index and a sentence index from that document for easy reference.

Classes:
    - DataProcessor:
        A class that processes and stores occurrence information about words in multiple documents.

        Methods:
            - add_dir(dir) -> None:
                Adds all text files from a specified directory to the data processor instance.

            - add_file(path) -> None:
                Adds a file to the data processor's path list.

            - all_sentences():
                Returns a list of SentencePosition instances for all sentences that contain at least one non-zero word occurrence.

            - sentence_positions(word:str):
                Returns a list of SentencePosition instances for sentences that contain the specified word.

            - generate():
                Generates the occurrence dictionary mapping every unique word to its occurrence in each document and sentence.

            - check_word(word:str):
                Checks if a word is present in the occurrence dictionary.

            - sentence_at(sp:SentencePosition):
                Returns the specific sentence at the given SentencePosition.

            - occurrences(word:str) -> int:
                Returns the total number of occurrences of a word across all documents.

            - document_occurences(word:str, index:int) -> int:
                Returns the number of occurrences of a word in a specific document by index.

            - docs_occurances_list(word:str):
                Returns a list with the number of occurrences of a specific word in each document.

            - __str__()->str:
                Returns a string representation of the occurrence dictionary.

Instance Variables:
    - paths: List[str]
        List of file paths for the documents.

    - occur_dict: Dict[str, dict[int, npt.NDArray]]
        A nested dictionary where keys are words and values are dictionaries that map document indices to arrays representing word occurrence in sentences.

    - sentences: list
        A list where each element contains the set of unique tokens found in the corresponding sentence of all documents.

    - sentences_size: list
        A list holding the sentence count for each document.

Private Instance Variables:
    - word_count_in_each_doc: npt.NDArray
        An numpy array holding the count of words for each document.
"""

from dataclasses import dataclass
from collections import defaultdict
import os, re, string
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple


def sentencize(str) -> List[str]:
    return list(filter(None, re.split(r'\.\s+', str)))
    # return str.split('. ')

def tokenize(str) -> List[str]:
    return str.lower().translate(str.maketrans('', '', string.punctuation)).split()
    # return str.lower().replace('. ', ' ').split()

@dataclass
class SentencePosition:
    doc_index: int
    sentence_index: int

class DataProcessor:
    paths: List[str]
    occur_dict:Dict[str,Dict[int,npt.NDArray]]
    sentences:list
    def __init__(self, path=None) -> None:
        self.occur_dict = defaultdict(dict)
        self.sentences_size = list()
        self.paths = list()
        self.sentences = list()
        if path is not None:
            self.add_file(path)

    def add_dir(self, dir) -> None:
        for file in os.listdir(dir):
            self.add_file(os.path.join(dir,file))

    def add_file(self, path) -> None:
        self.paths.append(path)

    def all_sentences(self):
        return [SentencePosition(doc_key, sentence_i) for word in self.occur_dict for doc_key, array in self.occur_dict[word].items() for sentence_i,occur in enumerate(array) if occur != 0]

    def sentence_positions(self, word:str):
        return [SentencePosition(doc_key, sentence_i) for doc_key,array in self.occur_dict[word].items() for sentence_i,occur in enumerate(array) if occur != 0]

    def generate(self):
        self.word_count_in_each_doc = np.zeros(len(self.paths), np.uint16)
        for doc_index, path in enumerate(self.paths):
            with open(path) as file:
                data = file.read()
            sentences = sentencize(data)
            self.sentences_size.append(len(sentences))
            self.word_count_in_each_doc[doc_index] = len(tokenize(data))

            for token in set(tokenize(data)):
                # sentence count
                self.occur_dict[token][doc_index] = np.zeros(self.sentences_size[doc_index],np.uint8)

            self.sentences.append([])
            for i, sentence in enumerate(sentences):
                sentence_tokens = tokenize(sentence)
                # self.doc_wordcount_list[doc_index] += len(sentence_tokens)
                self.sentences[-1].append(set(sentence_tokens))
                for token in self.sentences[-1][-1]:
                    self.occur_dict[token][doc_index][i] = sentence_tokens.count(token)

    # development helpers
    def check_word(self, word:str):
        if word not in self.occur_dict:
            raise KeyError(f"Error: word \"{word}\" not found in this instance of DataProcessor.")

    def sentence_at(self, sp:SentencePosition):
        with open(self.paths[sp.doc_index]) as file:
            data = file.read()
        return sentencize(data)[sp.sentence_index]

    def occurences(self, word:str) -> int:
        self.check_word(word)
        return np.sum(np.concatenate(list(self.occur_dict[word].values())))

    def document_occurences(self, word:str, index:int) -> int:
        if index >= len(self.paths) or index < 0:
            raise IndexError(f"Error: index is not valid. valid indexes for this instance are between 0 and {len(self.paths)-1}.")
        self.check_word(word)
        try:
            return np.sum(self.occur_dict[word][index])
        except(KeyError):
            return 0

    def docs_occurances_list(self, word:str):
        self.check_word(word)
        output_arr = np.zeros(len(self.paths), np.uint16)
        for key,value in self.occur_dict[word].items():
            output_arr[key] = np.sum(value)
        return output_arr

    def __str__(self)->str:
        output = ""
        for key, value in self.occur_dict.items():
            output+=f"'{key}': {value}\n"
        return output

