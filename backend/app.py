import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import csv 

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
# json_file_path = os.path.join(current_directory, 'init.json')

coffee_fix_csv_path =  os.path.join(
    os.environ['ROOT_PATH'], 'data/coffee_fix.csv')

app = Flask(__name__)
CORS(app)

name_to_desc1 = {}

with open('data/coffee_fix.csv', 'r') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for r in csv_reader:
        if r['desc_1'] is not None: 
            name_of_blend = r['name']
            description1 = r['desc_1']
            name_to_desc1[name_of_blend]= description1
        
def basic_search(query):
    results = []
    for name, description in name_to_desc1.items():
        if query.lower() in description.lower():
            results.append({"coffee_name": name, "description": description})
    return results[:5]

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@ app.route("/coffee")
def coffee_search():
    text = request.args.get("title")
    return json.dumps(basic_search(text))


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)