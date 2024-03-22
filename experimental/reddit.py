import praw
from praw.models import MoreComments

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pprint

# This needs to be run the first time you run this script ever
# nltk.download('all')

reddit = praw.Reddit(
    client_id="your client id",
    client_secret="your client secret",
    user_agent="macos:findyourcoffee:v1.0.0 (by u/worm-dealer)"
)
print(reddit.read_only)

# Show listings (general): top_ten_results = [s for s in reddit.subreddit("coffee").new(limit=10)]

# Search a subreddit using a query:

def find_query_in_comments(query, comment):

    def find_query_in_comments_acc(reply):
        if isinstance(reply, MoreComments):
            return []

        results = []
        lower_text = reply.body.lower()
        lines = lower_text.split('. ')
        for line in lines:
            if query in line:
                results.append(line)

        for r in reply.replies:
            results += find_query_in_comments_acc(r)
        
        return results

    return find_query_in_comments_acc(comment)

def preprocess_text(text):

    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

# query issue: 'press coffee' -> 'french press coffee'
query = "modcup"
query = query.lower()
search_results = [s for s in reddit.subreddit("coffee").search(query=query)]

query_found = []
for submission in search_results:
    lower_text = submission.selftext.lower()
    lines = '. '.join(lower_text.split('\n')).split('. ')
    for line in lines:
        if query in line:
            query_found.append(line)

    submission.comments.replace_more(limit=None)
    for c in submission.comments:
        query_found += find_query_in_comments(query, c)

for i, result in enumerate(query_found):
    print(str(i+1) + ')', result)

analyzer = SentimentIntensityAnalyzer()

for text in query_found:
    processed_text = preprocess_text(text)
    sentiments = analyzer.polarity_scores(text)
    print(sentiments)