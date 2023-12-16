import numpy as np
from data_processor import tokenize
from math import log

class TFIDF:
    def __init__(self, dataFolderPath, fileCount):
        self.dataFolderPath = dataFolderPath
        self.fileCount = fileCount
        self.data = list() # entire corpus
        for i in range(fileCount):
            docData = list() # 1 doc
            with open(f"{dataFolderPath}/document_{i}.txt", encoding="utf8") as file:
                for paragraph in file:
                    paragraphData = np.array(tokenize(paragraph)) # 1 paragraph
                    docData.append(paragraphData)
            # docData = np.concatenate(docData)
            self.data.append(docData)
    
    def calculate_tf(self, term:str, docNumber:int):
        """
        TF = number of times the term appears in a document / total number of words in the document
        """
        flatData = np.concatenate(self.data[docNumber])
        return len(flatData[flatData == term]) / len(flatData)
    
    def calculate_idf(self, term:str):
        """
        IDF = log(number of the documents in the corpus / number of the documents in the corpus that contain the term)
        """
        log()
        pass
    def calculate_tfidf(self, term:str, docNumber:int):
        """
        TF-IDF = TF * IDF
        """
        return calculate_tf(term, docNumber) * calculate_idf(term)

if __name__ == "__main__":
    tfidf = TFIDF("data", 1)
    term = input()
    for file in tfidf.data:
        print(type(file))
        print(file)

    print(tfidf.calculate_tf(term, 0))
            