import numpy as np

class TFIDF:
    def __init__(self, dataFolderPath, fileCount):
        self.dataFolderPath = dataFolderPath
        self.fileCount = fileCount
        self.data = list()
        for i in range(fileCount):
            with open(f"{dataFolderPath}/document_{i}.txt", encoding="utf8") as file:
                for paragraph in file:
                    pass

    def calculate_tf(self, term:str, docNumber:int):
        pass
    def calculate_idf(seld, term:str):
        pass