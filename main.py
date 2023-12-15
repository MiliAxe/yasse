#!/bin/python3
from search_engine import SearchEngine
from argparser import parser

if __name__ == "__main__":
    args = parser.parse_args()
    se = SearchEngine(args.files)
    se.calculate_tf_idf()

