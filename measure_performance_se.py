import sys
import time

from modules.bcolors import bcolors
from modules.searchengine import SearchEngine
from modules.dataprocessor import SentencePosition


# testing the speed
file_count = 50000
time_limit = 100
find_cosine_similarities = 0
try:
    file_count = int(sys.argv[1])
    find_cosine_similarities = int(sys.argv[2])
    time_limit = float(sys.argv[3])
except IndexError:
    pass
start_time = time.time()

paths = []
for i in range(file_count):
    paths.append(f"data/document_{i}.txt")
se = SearchEngine(paths)

se.calculate_tf_idf_all_docs()

if find_cosine_similarities:
    query = "microsoft rba"

    relevant_docs = sorted(se.cosine_similarities_docs(query), key=lambda x: x[2], reverse=True)
    print(relevant_docs[0])
    relevant_sentences = sorted(se.cosine_similarities_doc_sentences(query, relevant_docs[0][0]),
                                key=lambda x: x[1], reverse=True)
    print(relevant_sentences[0])
    print(
        f"relevant sentence: {se.dp.sentence_at(SentencePosition(relevant_docs[0][0], relevant_sentences[0][0]))}")

    highest_tf_word = se.get_word_with_highest_tf_in_doc(relevant_docs[0][0])
    highest_idf_word = se.get_top_five_idf_words_in_doc(relevant_docs[0][0])
    print(f"highest tf word: {highest_tf_word}")

end_time = time.time()
exec_time = end_time - start_time

print(f"Adding {bcolors.BLUE}{file_count}{bcolors.ENDC} files took {bcolors.BLUE}{exec_time:.3f}{bcolors.ENDC} seconds",
      end=" ")

if find_cosine_similarities:
    print("(Cosine similarities were calculated too)")
else:
    print()
try:
    assert (exec_time < time_limit)
    print(bcolors.GREEN + bcolors.BOLD + "Speed seems good!" + bcolors.ENDC)
except AssertionError:
    print(bcolors.RED + "Speed test failed! 💀" + bcolors.ENDC)
    exit(1)