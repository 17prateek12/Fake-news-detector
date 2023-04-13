from flask import Flask, jsonify, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import _stop_words
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.tokenize import word_tokenize
import numpy as np


model=pickle.load(open("fakemodel.pkl",'rb'))
cvec=pickle.load(open("cvec.pkl", 'rb'))

########with open("cvec.pkl", 'rb') as f:
 #   cvec = pickle.load(f)


app= Flask(__name__)



@app.route('/', methods=['GET'])
def hello_world():
    return render_template("prediction.html")

@app.route('/')
def prediction(text):
    text = text.str.replace('[^\w\s]',' ')
    text = re.sub(r'\d+', '', text)
    text = text.str.replace('[^A-Za-z]',' ')
    text = text.str.replace('  ',' ')
    text = text.str.replace('  ',' ')
    text = text.str.lower()
    stop_words = set(_stop_words.words('english'))
    #tokens = word_tokenize(text)
    filtertext = [word for word in text if word not in stop_words]
    pretext = ' '.join(filtertext)
    review_vect = cvec.transform([pretext]).toarray()
    predicted_label = model.predict(review_vect)
    if predicted_label == 0:
        prediction = "FAKE" 
    else:
        prediction= "REAL"
    return prediction

 
@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    #text_transformed = cvec.transform([text]).toarray()
    review_vect = cvec.transform([text]).toarray()
    review_vect = np.random.rand(232020)
    review_vect_2d = review_vect.reshape(1, -1)
    predicted_label = model.predict(review_vect_2d)
    if predicted_label == 0:
        prediction = "FAKE" 
    else:
        prediction= "REAL"
    print(prediction)
    return render_template('prediction.html', text=text, result="NEWS head line is {}".format(prediction))


@app.route('/prediction/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    review_vect = cvec.transform([text]).toarray()
    review_vect = np.random.rand(232020)
    review_vect_2d = review_vect.reshape(1, -1)
    predicted_label = model.predict(review_vect_2d)
    if predicted_label == 0:
        prediction = "FAKE" 
    else:
        prediction= "REAL"
    return jsonify(result=prediction)   
    


if __name__== '__main__':
    app.run()