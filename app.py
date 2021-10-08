from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
  return render_template('index.html')

@app.route("/get-recommendations/<username>", methods=['GET'])
def getRecommendations(username):
  print('---- USERNAME ', username);
  return "Return recommendations here!"

# run app
app.run(debug=True)

# <!-- FLASK_APP=app.py FLASK_ENV=development flask run -->