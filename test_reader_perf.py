# test_reader_perf.py

import unittest
import time
import threading
import tempfile
import csv
import os
import psutil
import tracemalloc
from data_curation import Reader

class TestReaderPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.large_csv_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
        writer = csv.writer(cls.large_csv_file)
        writer.writerow(['created_utc', 'author', 'skip', 'full_link', 'id', 'score', 'num_comments', 'selftext', 'title'])

        # Simulate 10000 rows
        for i in range(10000):
            author = f"User{i%500}"  # simulate 500 unique authors
            selftext = f"This is a test post about feeling very sad and hopeless number {i}."
            title = f"Post {i}"
            writer.writerow(['2023-01-01', author, '', f'link{i}', f'id{i}', '10', '5', selftext, title])
        cls.large_csv_file.close()

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.large_csv_file.name)

    def setUp(self):
        self.reader = Reader(self.large_csv_file.name)

    def test_getData_performance(self):
        start = time.time()
        tracemalloc.start()
        self.reader.getData()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = (time.time() - start) * 1000  # ms

        print(f"[getData] Time: {elapsed:.2f} ms, Memory Used: {current / 1024:.2f} KB, Peak: {peak / 1024:.2f} KB")

        self.assertEqual(self.reader.totalPosts(), 10000)
        self.assertGreaterEqual(len(self.reader.uniqueAuthors()), 500)

    def test_extractTopKeywords_under_load(self):
        self.reader.getData()
        start = time.time()
        keywords = self.reader.extractTopKeywordsFromAuthorPosts("User5", top_n=20)
        elapsed = (time.time() - start) * 1000

        print(f"[extractTopKeywords] Time: {elapsed:.2f} ms — Top: {keywords[:5]}")
        self.assertGreater(len(keywords), 0)

    def test_concurrent_getPostsByAuthor(self):
        self.reader.getData()

        def worker(author):
            return self.reader.getPostsByAuthor(author)

        authors = [f"User{i}" for i in range(10)]
        threads = []
        start = time.time()

        for a in authors:
            thread = threading.Thread(target=worker, args=(a,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        elapsed = (time.time() - start) * 1000
        print(f"[Concurrent Access] 10 threads completed in {elapsed:.2f} ms")

    def test_cpu_and_memory_usage(self):
        self.reader.getData()
        process = psutil.Process(os.getpid())
        cpu = process.cpu_percent(interval=1)
        mem = process.memory_info().rss / (1024 * 1024)  # in MB

        print(f"[System Stats] CPU Usage: {cpu:.2f}%, Memory Usage: {mem:.2f} MB")

        self.assertLess(cpu, 100)

    def test_binary_search_accuracy_and_speed(self):
        words = sorted(["sad", "happy", "depressed", "angry", "content", "joyful"])
        found_words = []
        start = time.time()
        for _ in range(10000):
            if self.reader.binary_search(words, "sad"):
                found_words.append("sad")
        elapsed = (time.time() - start) * 1000
        print(f"[Binary Search] 10k searches completed in {elapsed:.2f} ms")
        self.assertEqual(len(found_words), 10000)

if __name__ == '__main__':
    unittest.main()
