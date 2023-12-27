#!/bin/python3
from modules.argparser import parser
from modules.searchengine import SearchEngine, SentencePosition

if __name__ == "__main__":
    args = parser.parse_args()
    se = SearchEngine(args.files)
    se.calculate_tf_idf_all_docs()
    if args.query != "":
        relevant_docs = sorted(se.cosine_similarities_docs(args.query), key=lambda x: x[2], reverse=True)
        print(relevant_docs[0])
        relevant_sentences = sorted(se.cosine_similarities_doc_sentences(args.query, relevant_docs[0][0]),
                                    key=lambda x: x[1], reverse=True)
        print(relevant_sentences[0])
        print(
            f"relevant sentence: {se.dp.sentence_at(SentencePosition(relevant_docs[0][0], relevant_sentences[0][0]))}")

        highest_tf_word = se.get_word_with_highest_tf_in_doc(relevant_docs[0][0])
        highest_idf_word = se.get_top_five_idf_words_in_doc(relevant_docs[0][0])
        print(f"highest tf word: {highest_tf_word}")
        print(f"highest idf words: {highest_idf_word}")
