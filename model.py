import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

  def get_top_5_recommendations_3(self, username):
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

  def get_top_5_recommendations_2(self, username):
    """
    This function applies vectorizer on all 20 recommended products in one go inside a dataframe & then uses group by on product with mean() on sentiment score to get the top 5 products.
    NOTE: Although this works locally, it gives a memory exceeded error on heroku because data operations are happening on one large dataframe.
    To fix this issue, we are using a loop in get_top_5_recommendations method to avoid having one large data frame with all the data, and instead have small data batches.
    """

    # 1. Get top 20 recommendations
    top_20_product_recommendations = self.recommendation_system.loc[username].sort_values(ascending=False)[0:20]
    top_20_product_recommendations_list = top_20_product_recommendations.index.tolist()
    print('top_20_product_recommendations_list :')
    print(top_20_product_recommendations_list)
    
    # 2. Get top 20 products & their reviews in a dataframe
    top_products_reviews = self.og_ratings_df[self.og_ratings_df.name.isin(top_20_product_recommendations_list)][['name', 'reviews_title', 'reviews_text']]

    # 3. Pass the reviews to verctorizer
    print("Vector Transforming")
    tfidf_model = self.vectorizer_pickle.transform(top_products_reviews['reviews_text'])
    tfidf_df = pd.DataFrame(tfidf_model.toarray(), columns = self.vectorizer_pickle.get_feature_names())

    # 4. Make Predictions
    cutoff = 0.45
    print("Making Prediction")
    top_products_reviews["pos_sentiment_prob"] = self.sentiment_analysis_model.predict_proba(tfidf_df)[:, 1]
    top_products_reviews["sentiment_pred"] = top_products_reviews['pos_sentiment_prob'].map(lambda x: 1 if x >= cutoff else 0)

    # Group Prodcts and sort by sentiment scores
    top_products_list = top_products_reviews.groupby('name')['sentiment_pred'].mean().sort_values(ascending=False).index.tolist()
    print("top_products_list :")
    print(top_products_list)

    # Return top 5 results
    return top_products_list[:5]

  def get_top_5_recommendations(self, username):
    """
    In this method, 
      1. we get the top 20 receommendation from the recommendation system.
      2. We loop over all 20 products, get their reviews and use sentiment analysis model to predict the sentiment.
      3. We sort the products based on sentiment score and return top 5 results from 20.
    """

    # 1. Get top 20 recommendations
    top_20_product_recommendations = self.recommendation_system.loc[username].sort_values(ascending=False)[0:20]
    top_20_product_recommendations_list = top_20_product_recommendations.index.tolist()
    print('top_20_product_recommendations_list :')
    print(top_20_product_recommendations_list)

    sentiment_score_list = []

    # 2. Loop over all products, vectorize their reviews and get a sentiment score
    print("Starting Loop")
    for product in top_20_product_recommendations_list:
      print("Vector Transforming")
      products_reviews_df = self.og_ratings_df[self.og_ratings_df.name == product][['name', 'reviews_title', 'reviews_text']]
      reviews_tfidf_model = self.vectorizer_pickle.transform(products_reviews_df['reviews_text'])
      reviews_tfidf_df = pd.DataFrame(reviews_tfidf_model.toarray(), columns = self.vectorizer_pickle.get_feature_names())

      cutoff = 0.45
      print("Making Prediction")
      sentiment_prob_list = self.sentiment_analysis_model.predict_proba(reviews_tfidf_df)[:, 1]
      sentiment_pred_list = list(map(lambda x: 1 if x >= cutoff else 0, sentiment_prob_list));
      sentiment_score = sum(sentiment_pred_list)/len(sentiment_prob_list);
      sentiment_score_list.append(sentiment_score)

    print('Loop ended')

    # 3. Create a df with products and their sentiment score.
    product_sentiment_score_df = pd.DataFrame(data = {
      "product": top_20_product_recommendations_list,
      "sentiment_score": sentiment_score_list,
    })

    # 4. Get top 20 products list sorted
    top_products_list = product_sentiment_score_df.sort_values(by="sentiment_score", ascending=False)['product'].tolist()
    print('top_products_list')
    print(top_products_list)

    # Return top 5 products
    return top_products_list[:5]
