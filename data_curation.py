import csv
from collections import Counter
import re
from textblob import TextBlob
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import plotly.express as px

class Reader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {
            "created_utc": [],
            "author": [],
            "full_link": [],
            "id": [],
            "score": [],
            "num_comments": [],
            "selftext": [],
            "title": []
        }

    def getData(self):
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                if len(row) < 8: # only gets 8 rows
                    continue
                self.data["created_utc"].append(row[0])
                self.data["author"].append(row[1])
                self.data["full_link"].append(row[3])
                self.data["id"].append(row[4])
                self.data["score"].append(row[5])
                self.data["num_comments"].append(row[6])
                self.data["selftext"].append(row[7].replace("\n", "").replace("'", "'").strip()) # makes newline into space and replaces ' with ' because they weren't showing up properly
                self.data["title"].append(row[-1]) # gets the last or reverse index of the data and assigns it to title. (wasn't working when i assigned it as 8 so i decided to go reverse to get it)

        return self.data

    def dataSetInfo(self):
         
        print(f"\nAverage Text Length: {self.averagePostLength():.2f} words")

        print("\nTotal Posts:", self.totalPosts())

        print("\nUnique Authors:", len(self.uniqueAuthors()))

        print("\n")
        
        top_authors = self.topAuthors(10)
        current_rank = 1
        for author, amount_of_posts in top_authors:
            print(f"{current_rank}. {author} — {amount_of_posts} posts")
            current_rank += 1

    def averagePostLength(self):
        total_words = 0
        for post in self.data['selftext']:
            total_words += len(post.split())
        return total_words / self.totalPosts()

    def uniqueAuthors(self):
        return set(self.data["author"])
    
    def topAuthors(self, n): # Could've sorted this maunally, revese the sort to get the top authors and their counts. Counter is more readable
        return Counter(self.data["author"]).most_common(n)
    
    def totalPosts(self):
        return len(self.data["author"])
    
    # Binary Search for more opitmal searching in later methods
    def binary_search(self,arr, target):
        low = 0
        high = len(arr) - 1
        
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == target:
                return True
            elif arr[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return False

    def getPostsByAuthor(self, author): #You could also use (self, author, n) to get the top n posts by that author and not the total
        posts = [] # List containing all the posts from that author
        total_amount_of_posts = self.totalPosts()
        
        for post_index in range(total_amount_of_posts):
            if self.data["author"][post_index] == author:
                title = self.data["title"][post_index] # Get the title of the post
                selftext = self.data["selftext"][post_index] # The post itself
                posts.append((post_index, title, selftext)) # and append that to the post_index
        return posts


    # Filters the authors post by parameter placed keywords that can be placed inside of a list. this will go through that list of keywords and check
    # if the word exists inside of the authors also placed in the parameters post. you can place multiple authors and multiple keywords 
    def filterAuthorPostsByKeywords(self, author, keywords):
        posts = self.getPostsByAuthor(author)
        filtered = []
        lowercase_keywords = []
        
        for k in keywords:
            lowercase_keywords.append(k.lower())

        for post in posts:
            index, title, selftext = post
            combined_text = (title + " " + selftext).lower()
            
            cleaned_text = combined_text.replace('.', ' ')
            cleaned_text = cleaned_text.replace(',', ' ')
            cleaned_text = cleaned_text.replace('!', ' ')
            cleaned_text = cleaned_text.replace('?', ' ')
            words = cleaned_text.split()
            words.sort()

            for keyword in lowercase_keywords:
                if self.binary_search(words, keyword): # uses binary search for more optimal searching instead of O(n) search time
                    filtered.append((index, title, selftext))
                    break

        return filtered


    # Goes through the author's posts and extracts the top keywords used by that author by incrementing the top used keywords using word_count to keep track
    # of the words counter and getPostsByAuthor to keep track of the current authors post
    def extractTopKeywordsFromAuthorPosts(self, author, top_n = 100):

        stopwords = {
            "the", "and", "is", "in", "it", "try", "went", "bit", "long", "hear", "took", "away", "of", "others", "bit"
            "to", "a", "i", "on", "for", "with", "that", "this", "was", "but", "are", "not", "have", "my", "be", "so", "they",
            "me", "you", "just", "at", "or", "if", "we", "from", "as", "an", "had", "like",
            "has", "do", "about", "because", "how", "what", "when", "can", "get", "your",
            "would", "could", "should", "don't", "can't", "i'm", "it's", "they're", "there",
            "their", "them", "then", "than", "will", "out", "up", "down", "off", "back", "much",
            "many", "more", "most", "also", "into", "through", "her", "him", "his", "she", "he", "we're",
            "been", "being", "did", "does", "doing", "were", "wasn't", "weren't", "isn't", "aren't", "hasn't",
            "hadn't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "i'd", "you'd", "we'd", "they'd",
            "i've", "you've", "we've", "they've", "i'll", "you'll", "we'll", "they'll", "let's", "that's", "who's",
            "what's", "here", "there's", "ain't", "y'all", "yo", "nah", "yeah", "uh", "uhh", "uhm", "hmm", "lol", 
            "okay", "ok", "alright", "maybe", "even", "still", "some", "any", "each", "every", "ever",
            "always", "never", "often", "sometimes", "again", "already", "yet", "very", "really", "too",
            "no", "yes", "such", "am", "via", "per", "around", "between", "under", "over", "before", "after",
            "while", "during", "once", "twice", "although", "though", "where", "why", "whose", "whom", "which",
            "either", "neither", "both", "all", "none", "one", "two", "three", "first", "second", "next", "last", "i'm"
            "it’s", "you're", "we're", "they're", "he's", "she's", "that's", "what's", "who's", "don't", "want", "need", 
            "else", "know","today","tomorrow", "yesterday", "now", "then", "here", "there", "wherever", "whenever", "whereas",
            "time", "trying", "getting","bed","honestly","really", "seriously", "literally", "basically", "actually", "probably",
            "definitely", "maybe", "kinda", "sorta", "like", "just", "wish", "rest", "people", "anyone", "told", "got", "make",
            "lay", "things", "everyone", "look", "tells", "see", "saw", "looked", "looking", "keeps", "feel", "felt", "feeling",
            "nice", "turn", "head", "same", "cant", "rant", "made", "make", "makes", "making", "made", "made", "made", "made", "sub", "pull",
            "sure", "having", "break", "someone", "these", "its", "bench", "sitting", "dealing", "wants", "keep", "ill", "pass", "free", "talk"
            "talking", "talked", "talks", "say", "said", "says", "said", "said", "said", "said", "said", "said", "said", "deleted", "anything",
            "anymore", "tell", "thought", "new", "may", "place", "dont", "come", "yourself", "guys", "body", "give", "thing", "think", "who", "myself",
            "something", "done", "work", "job", "enough", "right", "live", "way", "only", "please", "year", "self","other", "part", "take", "person", 
            "hours", "college", "call", "past", "week", "covid", "whole", "months", "late", "reason", "school", "eat", "another", "support", "idk", 
            "years", "effort", "doesn't", "world", "parents", "internet", "stay", "dog", "talk", "going", "idea", "called", "everything", "similar",
            "starting","gonna","constantly","incredibly", "absolutely", "definitely", "probably", "maybe", "kinda", "sorta", "like","wanted","tonight","figure",
            "well", "amount","possible", "realised", "supposed", "rock", "post", "fine","thinking", "pretty", "hobbies","real","since","country","pretty","understand",
            "find", "longer", "night", "guy", "girl", "writing", "enjoy", "good", "better", "life", "pay", "become", "might", "push", "random", "easier", "tried",
            "thoughts", "happy", "posts", "chest", "day", "walk", "finally", "path", "inside", "our", "hope","apart","heard","believe","story", "loat","words",
            "dirt", "those", "exercise", "reinventing", "upon", "own", "far", "wall", "wishing", "wonder", "bring", "sit", "soul", "comfortable","glad", "itll",
            "staying","wait", "doesnt", "pool", "cleaning", "interests", "etc", "soon", "put", "care", "peace", "theres", "gym", "experience", "completely",
            "choose","instead"
        }

        word_count = Counter()
        posts = self.getPostsByAuthor(author)

        for post in posts:
            index, title, selftext = post
            combined_text = (title + " " + selftext).lower()


            combined_text = combined_text.replace("’", "'")


            cleaned_text = re.sub(r"[^\w\s']", " ", combined_text)
            words = cleaned_text.split()

            for word in words:
                clean_word = word.strip().lower()
                if clean_word not in stopwords and len(clean_word) > 2:
                    word_count[clean_word] += 1

        return word_count.most_common(top_n)


    # goes through the authors data and creates a blob. that blob will ccheck the polarity or how positive or negative the post is and 
    # the subjectivity of the post. it'll than return the word count as well and create a vector and vectorize these datas for it to be used in a K-Means
    # cluster
    def computeSentimentVectors(self):
        author_data_sentiment = {}
        for i in range(len(self.data["author"])): 
            author = self.data["author"][i]
            title_and_self_text = self.data["title"][i] + " " + self.data["selftext"][i]
            blob = TextBlob(title_and_self_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            word_count = len(title_and_self_text.split())

            if author not in author_data_sentiment:
                author_data_sentiment[author] = []
            author_data_sentiment[author].append((polarity, subjectivity, word_count))

        author_vectors = {}
        for author, sentiments in author_data_sentiment.items():
            sentiments = np.array(sentiments)
            mean_vector = sentiments.mean(axis = 0)
            author_vectors[author] = mean_vector

        return author_vectors


    # Takes the vectoized polarity, subjectvitiy and word count and returns its cluster state using k-means cluster algorithm; Fitting each single vector 
    # to a corresponding label this lebel will be used to plot the clusters on a cartesian graph. we can implement more label groups or clusters as needed
    def clusterAuthors(self, author_vectors, n_clusters = 3):
        authors = list(author_vectors.keys())
        vectors_list = []
        for a in authors:
            vectors_list.append(author_vectors[a])
        vectors = np.array(vectors_list)


        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(vectors)
        df = pd.DataFrame(vectors, columns=['polarity', 'subjectivity', 'length'])
        df['author'] = authors
        df['cluster'] = labels
        return df

    # plots the polarity as the x-axis the subjectvity as the y-axis and we can hover over each point and see the name and length of the cluster as well as 
    # assigning each cluster or label with its own color. 
    def plotClustersOnCartesianGraph(self, df):
        fig = px.scatter(
            df,
            x='polarity',
            y='subjectivity',
            color=df['cluster'].astype(str),
            hover_name='author',
            hover_data=['length'], 
            title="Authors and their Self-Text Clustered by Sentiment"
        )
        fig.show()


def main():
    file_path = 'depression-sampled.csv'
    author_name = 'What_I_do_45'
    keywords = ['help', 'sad', 'depressed']

    reader = Reader(file_path)
    reader.getData()

    print("Dataset Summary")
    reader.dataSetInfo()


# Gets the first n posts from a given author
    
    print(f"\nPosts by Author: {author_name}")
    posts = reader.getPostsByAuthor(author_name)
    for i, (index, title, selftext) in enumerate(posts[:5]):  # print first 3 posts
        print(f"\nPost #{i+1}")
        print("Title:", title)
        print(f"Self-Text: {selftext[:400]}...")  # First 400 characters of the selftext

# Gets all the posts contianing the key words from this author the first 150 characters from the self text body

    print(f"\nPosts by {author_name} containing these keywords {keywords}")
    filtered = reader.filterAuthorPostsByKeywords(author_name, keywords)
    for i, (index, title, selftext) in enumerate(filtered):
        print(f"\nMatched Post #{i+1}")
        print("Title:", title)
        print("Body:", selftext[:150], "...")

# Gets the top keywords used by this author ( first 100 words )
    print(f"\n Top Keywords Used by {author_name}")
    top_keywords = reader.extractTopKeywordsFromAuthorPosts(author_name, top_n = 100)
    for word, freq in top_keywords:
        print(f"{word}: {freq}")


    # creates the vectors from the object reader and then creates n = __ clusters for how many clusters you want to create and finally puts it all onto
    # the carteisan graph to show 
    vectors = reader.computeSentimentVectors()
    df_clusters = reader.clusterAuthors(vectors, n_clusters = 3)
    reader.plotClustersOnCartesianGraph(df_clusters)


if __name__ == "__main__":
    main()
