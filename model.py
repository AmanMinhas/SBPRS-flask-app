import pandas as pd
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class RecommendationSystem():
  def __init__(self):
    self.og_ratings_df = pd.read_csv("./data/sample30.csv")

    # load models
    recommendation_pickle_filepath = './models/user_final_rating.pickle'
    sentiment_analysis_mdel_filepath = './models/lr_sentiment_analysis_model.pickle'
    vectorizer_pickle_filepath = './models/tfidf_vectorizer.pickle'

    self.recommendation_system = pickle.load(open(recommendation_pickle_filepath, 'rb'))
    self.sentiment_analysis_model = pickle.load(open(sentiment_analysis_mdel_filepath, 'rb'))
    self.vectorizer_pickle = pickle.load(open(vectorizer_pickle_filepath, 'rb'))

    # Create a lemmatizer
    self.lemmatizer = WordNetLemmatizer()

  def document_preprocess(self, document):
    # convert to lowercase 
    document = document.lower()
    # tokenize into words
    words = word_tokenize(document)
    # replace "." with ". ". Add space after drop so that when puntuations are removed. 2 words are not grouped together.
    # TODO
    # replace puntuations with space
    for i in range (0, len(words)):
      word = words[i]
      words[i] = "".join(char if char not in set(string.punctuation) else " " for char in word)
    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    # lemmatize 
    words = [self.lemmatizer.lemmatize(word) for word in words]
    document = " ".join(words)
    return document

  def get_top_20_recommendations(self, username):
    # 1. Get top 20 recommendations
    top_20_product_recommendations = self.recommendation_system.loc[username].sort_values(ascending=False)[0:20]
    top_20_product_recommendations_list = top_20_product_recommendations.index.tolist()
    print('top_20_product_recommendations_list :')
    print(top_20_product_recommendations_list)
    
    # 2. Preprocess all reviews for top 20 products
    print("Proprocessing documents")
    top_products_reviews = self.og_ratings_df[self.og_ratings_df.name.isin(top_20_product_recommendations_list)][['name', 'reviews_title', 'reviews_text']]
    top_products_reviews['preprocessed_review'] = top_products_reviews.reviews_text.apply(self.document_preprocess)
    print("Proprocessing completed")

    # 3. Pass the preprocessed reviews to verctorizer
    print("Vector Transforming")
    tfidf_model = self.vectorizer_pickle.transform(top_products_reviews['preprocessed_review'])
    tfidf_df = pd.DataFrame(tfidf_model.toarray(), columns = self.vectorizer_pickle.get_feature_names())

    cutoff = 0.42
    print("Making Prediction")
    top_products_reviews["pos_sentiment_prob"] = self.sentiment_analysis_model.predict_proba(tfidf_df)[:, 1]
    top_products_reviews["sentiment_pred"] = top_products_reviews['pos_sentiment_prob'].map(lambda x: 1 if x >= cutoff else 0)

    top_products_list = top_products_reviews.groupby('name')['sentiment_pred'].mean().sort_values(ascending=False).index.tolist()
    print("top_products_list :")
    print(top_products_list)

    return top_products_list
