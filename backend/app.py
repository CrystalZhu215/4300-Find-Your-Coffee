import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
import csv
import findTop10
import SVD
from sklearn.feature_extraction.text import TfidfVectorizer

# RUN: flask run --host=0.0.0.0 --port=5000

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
# json_file_path = os.path.join(current_directory, 'init.json')

coffee_fix_csv_path = os.path.join(os.environ["ROOT_PATH"], "data/data_cleaning_coffee.csv")

app = Flask(__name__)
CORS(app)

name_to_desc1 = {}

with open("data/data_cleaning_coffee.csv", "r") as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for r in csv_reader:
        if r["desc_1"] is not None:
            name_of_blend = r["name"]
            description1 = r["desc_1"]
            name_to_desc1[name_of_blend] = description1

# Get sentiments
with open('sentiments.json') as f:
    sentiments = json.load(f)

# Get documents
df = pd.read_csv("data/data_cleaning_coffee.csv")
df['desc_all'] = df['desc_1'] + '\n' + df['desc_2'] + '\n' + df['desc_3']
df['desc_all'] = df['desc_all'].astype(str)

documents = df.values.tolist()

# Get relevance
query_to_relevant = {}
query_to_irrelevant = {}

# Get name to index
combined_names = df[["name"]].apply(lambda x: " ".join(x.dropna()), axis=1)
coffee_name_to_index = {
    name: i for i, name in enumerate(combined_names)
}

def basic_search(query):
    results = []
    for name, description in name_to_desc1.items():
        if query.lower() in description.lower():
            results.append({"coffee_name": name, "description": description})
    return results[:5]


def cosineSearch(query):
    results = findTop10.findTopTen(query)
    answers = []
    for i, x in enumerate(results):
        # print(x[1])
        answers.append(
            {
                "coffee_name": x[0]["name"],
                "description": x[0]["description"],
                "similarity score": x[1],
            }
        )
    return answers

def SVDSearch(query):
    if query not in query_to_relevant.keys():
        query_to_relevant[query] = df['name'].tolist()
        query_to_irrelevant[query] = []

    relevant = query_to_relevant[query]
    irrelevant = query_to_irrelevant[query]

    results = SVD.perform_SVD(documents, query, relevant, irrelevant, coffee_name_to_index)
    answers = []
    for _, name, roaster, desc, sim in results:
        answers.append(
            {
                "coffee_name": name,
                "roaster": roaster,
                "description": desc,
                "sim_score": sim,
            }
        )
    return answers

def rank(answers):

    for coffee in answers:

        roaster = coffee["roaster"].lower()

        if roaster in sentiments.keys() and len(sentiments[roaster]) > 0:

            avg_pos = 0
            avg_neg = 0
            comments = sentiments[roaster]

            for comment in comments:

                avg_pos += comment[1]["pos"]
                avg_neg += comment[1]["neg"]
                
            avg_pos /= len(comments)
            avg_neg /= len(comments)
            coffee["reddit_score"] = avg_pos - avg_neg

        else:

            coffee["reddit_score"] = 0

        if coffee["reddit_score"] == 0:
            coffee["social_score"] = "Neutral"
        elif coffee["reddit_score"] < 0:
            coffee["social_score"] = "{}% Negative".format(-round(coffee["reddit_score"] * 100, 2))
        else:
            coffee["social_score"] = "{}% Positive".format(round(coffee["reddit_score"] * 100, 2))

    answers = sorted(answers, key=(lambda x: x["reddit_score"]), reverse=True)

    return answers

@app.route("/")
def home():
    return render_template("base.html", title="sample html")

@app.route("/coffee")
def coffee_search():
    text = request.args.get("title")
    return json.dumps(cosineSearch(text))

@app.route("/coffee-SVD")
def coffee_SVD_search():
    query = request.args.get("title")

    answers = rank(SVDSearch(query))
    
    return json.dumps(answers)

@app.route('/relevance-update', methods=['POST'])
def feedback_submit():
    body = json.loads(request.data)

    query = body.get("title")
    coffee_name = body.get("coffee_name")
    isRelevant = body.get("relevant")

    print("data is here: ", query, coffee_name, isRelevant)

    if coffee_name in query_to_relevant[query] and isRelevant == False:
        query_to_irrelevant[query].append(coffee_name)
        query_to_relevant[query].remove(coffee_name)
    elif coffee_name in query_to_irrelevant[query] and isRelevant == True:
        query_to_relevant[query].append(coffee_name)
        query_to_irrelevant[query].remove(coffee_name)

    print("query:", query)
    print("relevant:", query_to_relevant[query])
    print("irrelevant:", query_to_irrelevant[query])

    return 'SUCCESS'

if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=8000)
