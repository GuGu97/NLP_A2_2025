import os
import re
import string
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from bs4 import BeautifulSoup


API_KEY = os.getenv("stack_app_key")
BASE_URL = "https://api.stackexchange.com/2.3"
SITE = "stackoverflow"
OUTPUT_FILE = "data/dataset.csv"


# Download questions with "nlp" tag, 100 questions per request
def fetch_questions(page):
    url = f"{BASE_URL}/search/advanced"

    params = {
        "page": page,
        "pagesize": 100,
        "order": "desc",
        "sort": "activity",
        "tagged": "nlp",
        "site": SITE,
        "filter": "withbody",
        # "accepted": "True",
        # "views": "30000",
        "key": API_KEY,
    }

    response = requests.get(url, params=params).json()
    return response


# Download answers for the given quesion ids, 100 answers per request
def fetch_answers(ids):
    ids_str = ";".join(map(str, ids))
    url = f"{BASE_URL}/answers/{ids_str}"

    params = {
        "pagesize": 100,
        "site": SITE,
        "filter": "withbody",
        "key": API_KEY,
    }

    response = requests.get(url, params=params).json()
    return response


# Download questions and answers, data size will be 100 * pages
def load_qa_data(pages):
    total_data = dict()

    for i in range(pages):
        qid_2_data = dict()
        a_id_set = set()

        # Fetch questions from the api
        q_json_list = fetch_questions(i + 1).get("items")

        # Reformat the questions
        for q_json in q_json_list:
            question = {
                "tags": q_json.get("tags"),
                "a_id": q_json.get("accepted_answer_id"),
                "q_id": q_json.get("question_id"),
                "date": q_json.get("creation_date"),
                "link": q_json.get("link"),
                "title": q_json.get("title"),
                "body": q_json.get("body"),
                "view_count": q_json.get("view_count"),
            }

            qid_2_data[question["q_id"]] = question

            # Questions that have an accepted answer
            if question["a_id"] != None:
                a_id_set.add(question["a_id"])

        # Fetch answers from the api
        a_json_list = fetch_answers(a_id_set).get("items")
        aid_2_answer = dict()

        # Reformat the answers
        for a_json in a_json_list:
            a_id = a_json.get("answer_id")
            answer = a_json.get("body")

            aid_2_answer[a_id] = answer

        # Match the answers with questions
        for _, question in qid_2_data.items():
            a_id = question["a_id"]
            if a_id != None:
                answer = aid_2_answer[a_id]
                question["answer"] = answer

        total_data.update(qid_2_data)
        time.sleep(0.3)

    return total_data


# Save the data to local csv file
def store_data():
    res = load_qa_data(200)
    df = pd.DataFrame.from_dict(res, orient="index")
    df.to_csv(OUTPUT_FILE)


# Preprocess the dataset, then rewrite it to the local file
def data_preprocess():
    data_df = pd.read_csv(OUTPUT_FILE, index_col=0)

    # Concatenate title and body
    data_df["cleaned_text"] = data_df["title"] + " " + data_df["body"]

    # Load the spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Do the text cleaning for only title
    data_df["cleaned_title"] = data_df["title"].apply(lambda x: text_clean(x, nlp))

    # Do the text cleaning for title and body
    data_df["cleaned_text"] = data_df["cleaned_text"].apply(
        lambda x: text_clean(x, nlp)
    )

    # Save cleaned data with UTF-8 encoding and no index name
    data_df.to_csv(
        OUTPUT_FILE, index=True, encoding="utf-8", na_rep="", index_label=None
    )

    return data_df


def get_data():
    # Load exactly the same as it was saved
    data_df = pd.read_csv(OUTPUT_FILE, index_col=0, encoding="utf-8")
    return data_df


# The data preprocessing workflow for one data row
def text_clean(text, nlp):
    soup = BeautifulSoup(text, "html.parser")
    for code_tag in soup.find_all("code"):
        code_tag.decompose()

    # Remove codes and html tags
    text = soup.get_text()
    # Remove urls
    text = re.sub(r"http\S+", "", text)
    # Remove '@' character
    text = re.sub(r"@", "", text)
    # Remove non-ASCII characters
    text = "".join([char for char in text if ord(char) < 128])
    # Remove newline characters
    text = text.replace("\n", " ").replace("\r", " ")
    # Lower the text
    text = text.lower()
    # Remove stop word and lemmatize
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    text = " ".join(tokens)
    # Remove punctuations
    text = text.translate(text.maketrans("", "", string.punctuation))
    # Remove extra spaces
    text = " ".join(text.split())

    return text
