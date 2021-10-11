from flask import Flask, render_template, jsonify
from model import RecommendationSystem

app = Flask(__name__)

@app.route("/")
def index():
  return render_template('index.html')

@app.route("/get-recommendations/<username>", methods=['GET'])
def getRecommendations(username):
  print('---- USERNAME ', username);
  recommendation_system = RecommendationSystem()
  recommended_products, error = recommendation_system.get_top_5_recommendations(username)

  if (error):
    err_response = jsonify({"error": error})
    err_response.status_code = 400
    return err_response

  return jsonify({"products": recommended_products})
# run app
# app.run(debug=True)

# <!-- FLASK_APP=app.py FLASK_ENV=development flask run -->