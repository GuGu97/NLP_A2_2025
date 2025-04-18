from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
import numpy as np


# LDA topic modelling with count vectors
def lda_with_count_vectors(data, topics, topk):
    # Cnovert text into count matrix
    # Ignore words appear in >85% of document to eliminate very common words
    lda_count_vectorizer = CountVectorizer(max_df=0.85)
    lda_count_matirx = lda_count_vectorizer.fit_transform(data["cleaned_text"])

    # Initialise LDA model to extract a given number of topics
    lda_count_model = LatentDirichletAllocation(n_components=topics, random_state=42)
    lda_count_model.fit(lda_count_matirx)

    # Retrieve the list of words in the vobulary
    feature_names_count = lda_count_vectorizer.get_feature_names_out()

    # Display the top k words for each topic
    top_words_list = []
    for topic in lda_count_model.components_:
        top_words = [feature_names_count[i] for i in topic.argsort()[-topk:]]
        top_words_list.append(top_words)

    return top_words_list


# LDA topic modelling with TF-IDF
def lda_with_tfidf_vectors(data, topics, topk):
    # Convert text into tf-idf matrix
    # Ignore words appear >85% in documents to eliminate very common words
    LDA_tf_vectorizer = TfidfVectorizer(max_df=0.85)
    lda_tfidf_matrix = LDA_tf_vectorizer.fit_transform(data["cleaned_text"])

    # Initialise LDA model to extract 15 topics
    lda_tf_model = LatentDirichletAllocation(n_components=topics, random_state=42)
    lda_tf_model.fit(lda_tfidf_matrix)

    # Retrieve the list of words in the vobulary
    feature_names_tf = LDA_tf_vectorizer.get_feature_names_out()

    # Display the top k words for each topic
    top_words_list = []
    for topic in lda_tf_model.components_:
        top_words = [feature_names_tf[i] for i in topic.argsort()[-topk:]]
        top_words_list.append(top_words)

    return top_words_list


# K-means clustering with tf-idf
def kmeans_with_tfidf_vectors(data, clusters, topk):

    # Convert text into tf-idf matrix
    # Ignore words appear >85% in documents to reduce very common words
    k_vectorizer = TfidfVectorizer(max_df=0.85)
    k_tfidf = k_vectorizer.fit_transform(data["cleaned_text"])

    # Initialise k-means clustering model with specified number of clusters
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(k_tfidf)

    # Get feature indices for each cluster center sorted in descending order
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = k_vectorizer.get_feature_names_out()

    # Display a given number of clusters and each with top k terms closest to centre
    top_words_list = []
    for i in range(clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :topk]]
        top_words_list.append(top_terms)

    return top_words_list


# Function to average the word vectors and produce a fixed-length document representation
def compute_doc_vectorise(docs, model):
    doc_embeddings = []
    for tokens in docs:
        # Collect vectors for tokens that exist in the model's vocabulary
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            # Compute the mean vector for the document
            avg_vec = np.mean(vectors, axis=0)
        else:
            # Return a zero vector if no tokens are found in the vocabulary
            avg_vec = np.zeros(model.vector_size)
        doc_embeddings.append(avg_vec)

    return np.array(doc_embeddings)


# K-means clustering with Word2Vec
def kmeans_with_word2vec(data, clusters, topk):
    # Tokenise and train a Wrod2Vec model on it
    tokenized_docs = [word_tokenize(doc) for doc in data["cleaned_text"]]
    k_word2vec = Word2Vec(tokenized_docs)

    vectorized_docs = compute_doc_vectorise(tokenized_docs, model=k_word2vec)

    # Initialise and fit the k-means clustering model
    kmeans_word2vec = KMeans(n_clusters=clusters)
    cluster_labels = kmeans_word2vec.fit(vectorized_docs)

    # Display 15 clusters and each with top 5 terms
    tokens_all_cluster = []

    for i in range(clusters):
        tokens_per_cluster = []

        # Get most representative words for all clusters
        most_representative = k_word2vec.wv.most_similar(
            positive=[cluster_labels.cluster_centers_[i]], topn=topk
        )
        for t in most_representative:
            tokens_per_cluster.append(t[0])

        tokens_all_cluster.append(tokens_per_cluster)

    return tokens_all_cluster


# create and fit BERTopic model, based on given docs and number of topics
def get_bert_topic(docs, n):
    topic_model = BERTopic(nr_topics=n)
    topics, probs = topic_model.fit_transform(docs)

    return topics, topic_model.get_topic_info()
