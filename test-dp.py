import unittest
from data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.dp = DataProcessor()
        self.DOC_SIZE = 4
        for i in range(self.DOC_SIZE):
            self.dp.add_file(f"tests/doc{i}")
        self.dp.generate()

    def test_docs_occurances_list(self):
        self.assertEqual(self.dp.docs_occurances_list("this").tolist(), [5, 1, 1, 0])
        self.assertEqual(self.dp.docs_occurances_list("the").tolist(), [0, 0, 2, 0])
        self.assertEqual(self.dp.docs_occurances_list("lorem").tolist(), [0, 1, 0, 5])
        self.assertEqual(self.dp.docs_occurances_list("test").tolist(), [2, 2, 0, 0])

    def test_sentence_at(self):
        self.assertEqual(self.dp.sentence_at(0, 1), "it should have no in this sentence")
        self.assertEqual(self.dp.sentence_at(3, 0), 
        "Lorem ipsum dolor sit amet, officia excepteur ex fugiat reprehenderit enim labore culpa sint ad nisi Lorem pariatur mollit ex esse exercitation amet")
        self.assertEqual(self.dp.sentence_at(2, -1), "even this has been only word that occurred in every sentence imaginable")

    def test_docs_words_count_list(self):
        dp_tmp = DataProcessor("data/document_1924.txt")
        dp_tmp.add_file("data/document_8666.txt")
        dp_tmp.generate()

        self.assertEqual(self.dp.word_count_in_each_doc.tolist(), [23, 22, 47, 105])
        self.assertEqual(dp_tmp.word_count_in_each_doc.tolist(), [607, 362])

    def test_add_data_file(self):
        self.assertEqual(self.dp.paths, [f"tests/doc{i}" for i in range(self.DOC_SIZE)])

    def test_check_word(self):
        self.dp.check_word("this")
        self.dp.check_word("test")
        self.dp.check_word("lorem")
        self.dp.check_word("the")
        with self.assertRaises(KeyError):
            self.dp.check_word("madeuplolword")

    def test_generation(self):
        tests = [(self.dp.occur_dict["lorem"],{1:[1, 0, 0], 3:[2, 0, 1, 0, 1, 1, 0, 0]}),
                 (self.dp.occur_dict["test"], {0:[1,0,1,0], 1:[0,1,1]}),
                 (self.dp.occur_dict["the"], {2:[0,2,0]}),
                 (self.dp.occur_dict["this"], {0:[2,1,1,1],1:[0,1,0],2:[0,0,1]}),]
        for actual,test in tests:
            # t sands for test, a stands for actual
            for adict, tdict in zip(actual.items(), test.items()):
                akey, aarray = adict
                tkey, tarray = tdict
                self.assertEqual(aarray.tolist(), tarray)
                self.assertEqual(akey, tkey)

    def test_document_occurences(self):
        self.assertEqual(self.dp.document_occurences("lorem", 1), 1)
        self.assertEqual(self.dp.document_occurences("lorem", 3), 5)
        self.assertEqual(self.dp.document_occurences("this", 3), 0)
        self.assertEqual(self.dp.document_occurences("this", 0), 5)

        with self.assertRaises(IndexError):
            self.dp.document_occurences("test",self.DOC_SIZE)

        with self.assertRaises(IndexError):
            self.dp.document_occurences("test",-1)

        with self.assertRaises(KeyError):
            self.dp.document_occurences("madeupword", 0)

    def test_occurences(self):
        self.assertEqual(self.dp.occurences("this"), 7)
        self.assertEqual(self.dp.occurences("the"), 2)
        self.assertEqual(self.dp.occurences("lorem"), 6)
        self.assertEqual(self.dp.occurences("test"), 4)

        with self.assertRaises(KeyError):
            self.dp.occurences("madeupword")



if __name__ == '__main__':
    unittest.main() 
