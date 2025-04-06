import os
import re
import string
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from bs4 import BeautifulSoup


class DataLoader:
    API_KEY = os.getenv("stack_app_key")
    BASE_URL = "https://api.stackexchange.com/2.3"
    SITE = "stackoverflow"
    OUTPUT_FILE = "data/dataset.csv"

    def fetch_questions(self, page):
        url = f"{self.BASE_URL}/search/advanced"

        params = {
            "page": page,
            "pagesize": 100,
            "order": "desc",
            "sort": "activity",
            "tagged": "nlp",
            "site": self.SITE,
            "filter": "withbody",
            # "accepted": "True",
            # "views": "30000",
            "key": self.API_KEY,
        }

        response = requests.get(url, params=params).json()
        return response

    def fetch_answers(self, ids):
        ids_str = ";".join(map(str, ids))
        url = f"{self.BASE_URL}/answers/{ids_str}"

        params = {
            "pagesize": 100,
            "site": self.SITE,
            "filter": "withbody",
            "key": self.API_KEY,
        }

        response = requests.get(url, params=params).json()
        return response

    def load_qa_data(self, pages):
        total_data = dict()

        for i in range(pages):
            qid_2_data = dict()
            a_id_set = set()
            q_json_list = self.fetch_questions(i + 1).get("items")

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

                if question["a_id"] != None:
                    a_id_set.add(question["a_id"])

            a_json_list = self.fetch_answers(a_id_set).get("items")
            aid_2_answer = dict()

            for a_json in a_json_list:
                a_id = a_json.get("answer_id")
                answer = a_json.get("body")

                aid_2_answer[a_id] = answer

            for _, question in qid_2_data.items():
                a_id = question["a_id"]
                if a_id != None:
                    answer = aid_2_answer[a_id]
                    question["answer"] = answer

            total_data.update(qid_2_data)
            time.sleep(0.3)

        return total_data

    def store_data(self):
        res = self.load_qa_data(200)
        df = pd.DataFrame.from_dict(res, orient="index")
        df.to_csv(self.OUTPUT_FILE)

    def data_preprocess(self):
        data_df = pd.read_csv(self.OUTPUT_FILE, index_col=0)

        # Concatenate title and body
        data_df["cleaned_text"] = data_df["title"] + " " + data_df["body"]

        # Load the spaCy English model
        nlp = spacy.load("en_core_web_sm")

        # Do the text cleaning
        data_df["cleaned_text"] = data_df["cleaned_text"].apply(
            lambda x: text_clean(x, nlp)
        )

        # Save the cleaned data
        data_df.to_csv(self.OUTPUT_FILE)

        return data_df


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
