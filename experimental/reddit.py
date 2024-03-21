from bs4 import BeautifulSoup
import requests
import praw
import pprint

reddit = praw.Reddit(
    client_id="your client id here",
    client_secret="your client secret here",
    user_agent="macos:findyourcoffee:v1.0.0 (by u/worm-dealer)"
)

print(reddit.read_only)
top_ten_results = [s for s in reddit.subreddit("coffee").new(limit=10)]
# pprint.pprint(vars(submission))

submission = top_ten_results[1]
print(submission.selftext)
for c in submission.comments:
    print('\n')
    print(c.body)
    # Need to expand comments

'''
def getdata(url):
    r = requests.get(url)
    return r.text

url = "https://www.reddit.com/r/coffee.json"
htmldata = getdata(url)
print(htmldata)

'''