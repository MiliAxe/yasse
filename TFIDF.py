import numpy as np
from data_processor import tokenize
from math import log
from jsonParser import jsonParser

class TFIDF:
    def __init__(self, dataFolderPath:str, candidates:list[int]):
        self.dataFolderPath = dataFolderPath
        self.candidates = candidates
        self.data = list() # entire candidates list
        # candidates = range(1)
        for candidate in candidates:
            docData = list() # 1 doc of candidates
            with open(f"{dataFolderPath}/document_{candidate}.txt", encoding="utf8") as file:
                for paragraph in file:
                    paragraphData = np.array(tokenize(paragraph)) # 1 paragraph of the doc
                    docData.append(paragraphData)
            # docData = np.concatenate(docData)
            self.data.append(docData)
    
    def calculate_tf(self, term:str, docNumber:int):
        """
        TF = number of times the term appears in a document / total number of words in the document
        """
        flatDoc = np.concatenate(self.data[docNumber])
        return len(flatDoc[flatDoc == term]) / len(flatDoc)
    
    def calculate_idf(self, term:str):
        """
        IDF = log(number of the documents in the corpus / number of the documents in the corpus that contain the term)
        """
        flatData = [np.concatenate(Doc) for Doc in self.data]
        Appeared = 0
        for Doc in flatData:
            if np.any(Doc == term):
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

    tfidf = TFIDF("data", range(10))
    term = input().lower()
    for file in tfidf.data:
        # for line in file:
        #     print(line[line == "the"])
        # print(file[file == "the"])
        # print(type(file))
        # print(file)
        pass
    print()
    # print(tfidf.calculate_tf(term, 0))
    # print(tfidf.calculate_idf(term))
    print(tfidf.calculate_tfidf(term, 0))
    