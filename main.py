#!/bin/python3
import time
from modules.searchengine import SearchEngine, SentencePosition
from modules.argparser import parser

if __name__ == "__main__":
    args = parser.parse_args()
    se = SearchEngine(args.files)
    se.calculate_tf_idf_all_docs()
    if args.query != "":
        relevant_docs = sorted(se.cosine_similarities_docs(args.query), key =lambda x: x[2], reverse=True)
        print(relevant_docs[0])
        relevant_sentences = sorted(se.cosine_similarities_doc_sentences(args.query, relevant_docs[0][0]), key =lambda x: x[1], reverse=True)
        print(relevant_sentences[0])
    
    
