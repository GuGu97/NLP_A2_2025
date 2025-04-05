import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt


class DataLoader:
    API_KEY = os.getenv("stack_app_key")
    BASE_URL = "https://api.stackexchange.com/2.3"
    OUTPUT_CSV = "so_qa_dataset.csv"
    SITE = "stackoverflow"

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
        df.to_csv("data/dataset.csv")
