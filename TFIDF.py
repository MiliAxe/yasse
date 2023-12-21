import numpy as np
from data_processor import tokenize
from math import log
from jsonParser import jsonParser
from collections import ChainMap

class TFIDF:
    def __init__(self, dataFolderPath:str, candidates:list[int]):
        self.dataFolderPath = dataFolderPath
        self.candidates = candidates
        self.data = list() # entire candidates list
        self.TFIDF_Data = list()
        
        # initialize data
        for candidate in candidates:
            docData = list() # 1 doc of candidates
            with open(f"{dataFolderPath}/document_{candidate}.txt", encoding="utf8") as file:
                for paragraph in file:
                    words = tokenize(paragraph)
                    """
                    Convert Data to dict later via:
                    dicted = {word: count for word in words for count in [words.count(word)]}
                    """
                    paragraphData = words # 1 paragraph of the doc (dict)
                    docData.append(paragraphData)
            # docData = np.concatenate(docData)
            self.data.append(docData)

        # initialize TD-IDF data
        self.TFIDF_Data = [sum(Doc, []) for Doc in self.data]
        self.TFIDF_Data = [{word: [count] for word in Doc for count in [Doc.count(word)]} for Doc in self.TFIDF_Data]
    def calculate_tf(self, term:str, docNumber:int):
        """
        TF = number of times the term appears in a document / total number of words in the document
        """
        words = sum(self.data[docNumber], [])
        flatDoc = {word: count for word in words for count in [words.count(word)]}
        return flatDoc[term] / len(flatDoc)
    
    def calculate_idf(self, term:str):
        # print(self.data)
        # for Doc in self.data:
        #     print(Doc)
        #     print("--------------------------------")
        """
        IDF = log(number of the documents in the corpus / number of the documents in the corpus that contain the term)
        """
        for flatDoc in self.TFIDF_Data:
            print(sorted([[key, value] for key, value in flatDoc.items()], key=lambda x: x[1], reverse=True))
            print("--------------------------------")

        Appeared = 0
        for Doc in self.data:
            Doc = sum(Doc, [])
            flatDoc = {word: count for word in Doc for count in [Doc.count(word)]}
            # print(sorted([[key, value] for key, value in flatDoc.items()], key=lambda x: x[1], reverse=True))
            # print("--------------------------------")
            if term in flatDoc.keys():
                Appeared += 1

        if (len(self.candidates) == 1 and Appeared == 1):
            return 1
        if Appeared == 0:
            return np.inf
        return np.log(np.divide(len(self.candidates), Appeared))

        
    def calculate_tfidf(self, term:str, docNumber:int):
        """
        TF-IDF = TF * IDF
        """
        TF = self.calculate_tf(term, docNumber)
        if TF == 0:
            return 0
        IDF = self.calculate_idf(term)
        return TF * IDF

if __name__ == "__main__":
    js = jsonParser("data.json")

    tfidf = TFIDF("data", range(1000))
    term = input().lower()
    for file in tfidf.data:
        # for line in file:
        #     print(line[line == "the"])
        # print(file[file == "the"])
        # print(type(file))
        # print(file)
        pass
    print()
    print(tfidf.calculate_tf(term, 0))
    # print(tfidf.calculate_idf(term))
    print(tfidf.calculate_tfidf(term, 0))
    