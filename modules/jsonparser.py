from dataclasses import dataclass
import ijson
from typing import List

@dataclass
class Documents:
    query:str
    candidates: List[int]
    selected: int|None

def index_parser(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return None

def json_return_docs(path)->List:
    with open(path, 'rb') as file:
        jsondata = ijson.items(file, 'item')
        return [Documents(document["query"], document["candidate_documents_id"], index_parser(document["is_selected"], 1)) for document in jsondata]

if __name__ == "__main__":
    print(json_return_docs("data.json"))
