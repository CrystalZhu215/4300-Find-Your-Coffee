import praw
from praw.models import MoreComments
import pprint

reddit = praw.Reddit(
    client_id="your client id",
    client_secret="your client secret",
    user_agent="macos:findyourcoffee:v1.0.0 (by u/worm-dealer)"
)
print(reddit.read_only)

# Show listings (general): top_ten_results = [s for s in reddit.subreddit("coffee").new(limit=10)]

# Search a subreddit using a query:

def print_if_query_found(query, comment):
    if isinstance(comment, MoreComments):
        return

    lower_text = comment.body.lower()
    if query in lower_text:
        lines = lower_text.split('\n')
        for line in lines:
            if query in line:
                print(line)
                print('-------------------------------------------------------------------------')

    for r in comment.replies:
        print_if_query_found(query, r)

# query issue: 'press coffee' -> 'french press coffee'
query = "Buon Caffe"
query = query.lower()
search_results = [s for s in reddit.subreddit("coffee").search(query=query)]

for submission in search_results:
    lower_text = submission.selftext.lower()
    if query in lower_text:
        lines = lower_text.split('\n')
        for line in lines:
            if query in line:
                print(line)
                print('-------------------------------------------------------------------------')

    submission.comments.replace_more(limit=None)
    for c in submission.comments:
        print_if_query_found(query, c)