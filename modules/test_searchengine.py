import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from concurrent.futures import ThreadPoolExecutor

import unittest
from modules.searchengine import SearchEngine
from jsonparser import json_return_docs
from random import random

def index_to_path(index):
    return f"data/document_{index}.txt"

def index_to_paths(indexes: list):
    return [index_to_path(index) for index in indexes]

def random_index(max_index):
    return int(random()*max_index)

def random_indexes(max_index, size):
    return [random_index(max_index) for _ in range(size)]


class TestSearchEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.testCases = json_return_docs("data.json")
        self.testCount = 10

    def get_search_result(self, query, candidates):
        se = SearchEngine(candidates)
        se.calculate_tf_idf_all_docs()
        
        relevant_docs = sorted(
            se.cosine_similarities_docs(query), key=lambda x: x[2], reverse=True
        )

        relevant_sentences = sorted(
            se.cosine_similarities_doc_sentences(query, relevant_docs[0][0]),
            key=lambda x: x[1],
            reverse=True,
        )

        return (relevant_docs, relevant_sentences)

    def test_search_all(self):
        for _ in range(self.testCount):
            current_testCase = self.testCases[random_index(len(self.testCases))]
            query = current_testCase.query
            candidates = index_to_paths(current_testCase.candidates)
            selected = current_testCase.selected


            relevant_docs, relevant_sentences = self.get_search_result(query, candidates)


            self.assertEqual(relevant_docs[0][0], len(current_testCase.candidates)-1, f"DOC RELEVANCE FAILED: query: {query}, test case: {current_testCase.candidates[-1]}")
            self.assertEqual(relevant_sentences[0][0], selected, f"SENTENCE RELEVANCE FAILED: query: \"{query}\", test case: {current_testCase.candidates[-1]}")

    def test_search_success_count(self):
        failures = 0
        for _ in range(self.testCount):
            try:
                current_testCase = self.testCases[random_index(len(self.testCases))]
                query = current_testCase.query
                candidates = index_to_paths(current_testCase.candidates)
                selected = current_testCase.selected

                relevant_docs, relevant_sentences = self.get_search_result(query, candidates)

                self.assertEqual(relevant_docs[0][0], len(current_testCase.candidates)-1, f"DOC RELEVANCE FAILED: query: {query}, test case: {current_testCase.candidates[-1]}")
                self.assertEqual(relevant_sentences[0][0], selected, f"SENTENCE RELEVANCE FAILED: query: \"{query}\", test case: {current_testCase.candidates[-1]}")
            except AssertionError:
                failures += 1

        if failures > 0:
            self.fail(f"{failures} out of {self.testCount} tests failed.")

    



if __name__ == "__main__":
    unittest.main()