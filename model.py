import pandas as pd
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class RecommendationSystem():
  def __init__(self):
    self.og_ratings_df = pd.read_csv("./data/sample30.csv")

    # load models
    recommendation_pickle_filepath = './models/user_final_rating.pickle'
    sentiment_analysis_mdel_filepath = './models/final_sentiment_analysis_model.pickle'
    vectorizer_pickle_filepath = './models/tfidf_vectorizer.pickle'
    product_sentiment_score_df_filepath = './models/product_sentiment_score_df.pickle'

    self.recommendation_system = pickle.load(open(recommendation_pickle_filepath, 'rb'))
    self.sentiment_analysis_model = pickle.load(open(sentiment_analysis_mdel_filepath, 'rb'))
    self.vectorizer_pickle = pickle.load(open(vectorizer_pickle_filepath, 'rb'))
    self.product_sentiment_score_df = pickle.load(open(product_sentiment_score_df_filepath, 'rb'))

    # Create a lemmatizer
    self.lemmatizer = WordNetLemmatizer()

  def get_top_5_recommendations_2(self, username):
    """
      This function uses the product_sentiment_score_df dataframe which was precreated from the notebook and uses it for sorting recommendations
    """
    # 1. Get top 20 recommendations
    top_20_product_recommendations = self.recommendation_system.loc[username].sort_values(ascending=False)[0:20]
    top_20_product_recommendations_list = top_20_product_recommendations.index.tolist()
    print('top_20_product_recommendations_list :')
    print(top_20_product_recommendations_list)

    # 2. Sort product list based on sentiment score 
    product_sentiment_score_df = self.product_sentiment_score_df
    top_20_products_sorted_df = product_sentiment_score_df[product_sentiment_score_df['product'].isin(top_20_product_recommendations_list)].sort_values(by="sentiment_score", ascending=False)
    top_20_products_sorted_list = top_20_products_sorted_df['product'].tolist()
    print('top_20_products_sorted_list')
    print(top_20_products_sorted_list)

    return top_20_products_sorted_list[:5]

  def get_top_5_recommendations(self, username):
    # 1. Get top 20 recommendations
    top_20_product_recommendations = self.recommendation_system.loc[username].sort_values(ascending=False)[0:20]
    top_20_product_recommendations_list = top_20_product_recommendations.index.tolist()
    print('top_20_product_recommendations_list :')
    print(top_20_product_recommendations_list)
    
    # 2. Preprocess all reviews for top 20 products
    # print("Proprocessing documents")
    top_products_reviews = self.og_ratings_df[self.og_ratings_df.name.isin(top_20_product_recommendations_list)][['name', 'reviews_title', 'reviews_text']]
    # top_products_reviews['preprocessed_review'] = top_products_reviews.reviews_text.apply(self.document_preprocess)
    # print("Proprocessing completed")

    # 3. Pass the preprocessed reviews to verctorizer
    print("Vector Transforming")
    # tfidf_model = self.vectorizer_pickle.transform(top_products_reviews['preprocessed_review'])
    tfidf_model = self.vectorizer_pickle.transform(top_products_reviews['reviews_text'])
    tfidf_df = pd.DataFrame(tfidf_model.toarray(), columns = self.vectorizer_pickle.get_feature_names())
    # tfidf_df = pd.DataFrame(tfidf_model.toarray(), columns = self.vectorizer_pickle.get_feature_names_out())

    cutoff = 0.45
    print("Making Prediction")
    top_products_reviews["pos_sentiment_prob"] = self.sentiment_analysis_model.predict_proba(tfidf_df)[:, 1]
    top_products_reviews["sentiment_pred"] = top_products_reviews['pos_sentiment_prob'].map(lambda x: 1 if x >= cutoff else 0)

    top_products_list = top_products_reviews.groupby('name')['sentiment_pred'].mean().sort_values(ascending=False).index.tolist()
    print("top_products_list :")
    print(top_products_list)

    return top_products_list[:5]
