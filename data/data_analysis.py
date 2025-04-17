from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize


# Generate wordcloud for the title in posts
def word_cloud_for_title(data):
    text = " ".join(word for word in data["cleaned_title"])
    wordc = WordCloud(width=800, height=600, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# Sort the terms based on the IDF scores (lowest first).
def sort_by_idf_for_title(data):
    vectorizer = TfidfVectorizer(max_df=0.85)
    tfidf = vectorizer.fit_transform(data["cleaned_title"])

    feature_names = vectorizer.get_feature_names_out()
    idf_scores = vectorizer.idf_

    # Create a dictionary mapping each term to its IDF score.
    idf_dict = dict(zip(feature_names, idf_scores))

    # Sort the terms based on the IDF scores (lowest first).
    sorted_terms = sorted(idf_dict.items(), key=lambda item: item[1])

    return sorted_terms


# Remove target words from the given text
def custom_stop_word_removal(text, words):
    tokens = word_tokenize(text)
    text = [word for word in tokens if word.lower() not in words]
    text = " ".join(text)
    return text


# Remove target words in titles
def word_removal_for_title(data, words):
    data["cleaned_title"] = data["cleaned_title"].apply(
        lambda x: custom_stop_word_removal(x, words)
    )
