from flask import Flask, render_template, jsonify
from model import RecommendationSystem

app = Flask(__name__)

@app.route("/")
def index():
  return render_template('index.html')

@app.route("/get-recommendations/<username>", methods=['GET'])
def getRecommendations(username):
  print('---- USERNAME ', username);
  # return "Return recommendations here!"
  recommendation_system = RecommendationSystem()
  # recommended_products = recommendation_system.get_top_5_recommendations(username)
  recommended_products = recommendation_system.get_top_5_recommendations(username)

  return jsonify({"products": recommended_products})
# run app
# app.run(debug=True)

# <!-- FLASK_APP=app.py FLASK_ENV=development flask run -->