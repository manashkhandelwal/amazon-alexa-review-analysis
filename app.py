from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load and process data
df = pd.read_csv('amazon_alexa.tsv', sep='\t')
df['verified_reviews'] = df['verified_reviews'].str.lower()

# Train model
X = df['verified_reviews']
Y = df['feedback']
cv = CountVectorizer()
X_transformed = cv.fit_transform(X)

logreg = LogisticRegression()
logreg.fit(X_transformed, Y)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    from flask import request
    review = request.form['review']
    transformed_review = cv.transform([review])
    prediction = logreg.predict(transformed_review)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
