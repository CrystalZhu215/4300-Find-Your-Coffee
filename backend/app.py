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
relevant = np.ones(len(documents))

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
    results = SVD.all_docs_to_query(query)
    answers = []
    for i, name, roaster, desc, sim in results:
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
    text = request.args.get("title")

    answers = rank(SVDSearch(text))
    
    return json.dumps(answers)

@app.route('/rocchio', methods=['POST'])
def feedback_submit():
    query = request.args.get('title')
    coffee_name = request.args.get("coffee_name")
    relevant = request.args.get('relevant')
    return jsonify("success"),200




df = pd.read_csv("data/data_cleaning_coffee.csv")
df['desc_all'] = df['desc_1'] + '\n' + df['desc_2'] + '\n' + df['desc_3']
df['desc_all'] = df['desc_all'].astype(str)
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 70, max_df = 0.7)
documents = df.values.tolist()
td_matrix = vectorizer.fit_transform([x[-1] for x in documents])



if "DB_NAME" not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=8000)
