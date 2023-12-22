#!/bin/python3
import time
from modules.searchengine import SearchEngine, SentencePosition
from modules.argparser import parser

if __name__ == "__main__":
    args = parser.parse_args()
    se = SearchEngine(args.files)
    se.calculate_tf_idf()
    se.sentence_tf_idf("ok")
