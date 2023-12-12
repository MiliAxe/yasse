import json


class jsonParser:
    def __init__(self, dataFilePath):
        with open(dataFilePath) as file:
            self.data = json.load(file)

    def getCandidateDocuments(self, documentID):
        return self.data[documentID]["candidate_documents_id"]

    def getQuery(self, documentID):
        return self.data[documentID]["query"]

    def getCorrectSentenceIndex(self, documentID):
        isSelectedArray = self.data[documentID]["is_selected"]

        for index in range(len(isSelectedArray)):
            if isSelectedArray[index] == 1:
                return index

        return -1
