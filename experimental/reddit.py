from bs4 import BeautifulSoup
import requests
import praw
from praw.models import MoreComments
import pprint

reddit = praw.Reddit(
    client_id="4E_NVygep-9QgyC9yf1Psg",
    client_secret="ZhEY7IYa9w-KkVDWf-RrrcOTeiIYGQ",
    user_agent="macos:findyourcoffee:v1.0.0 (by u/worm-dealer)"
)
print(reddit.read_only)

# Show listings

'''
top_ten_results = [s for s in reddit.subreddit("coffee").new(limit=10)]

submission = top_ten_results[1]
comment = submission.comments[0]
subcomment = comment.replies[0]

# To get attributes of an element: pprint.pprint(vars(comment))

print(submission.selftext)
print()
print(comment.body)
print()
print(subcomment.body)

'''

# Search a subreddit

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

query = 'modcup'
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

'''
def getdata(url):
    r = requests.get(url)
    return r.text

url = "https://www.reddit.com/r/coffee.json"
htmldata = getdata(url)
print(htmldata)

'''