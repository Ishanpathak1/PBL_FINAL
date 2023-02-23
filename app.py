from flask import Flask, render_template, request
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from collections import Counter

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        sentences = sent_tokenize(text)
        words = [word_tokenize(sentence.lower()) for sentence in sentences]
        stop_words = set(stopwords.words('english'))
        sentence_count = len(sentences)
        filtered_words = [[word for word in sentence if word not in stop_words] for sentence in words]
        tokenized_words = [' '.join(sentence) for sentence in words]
        filtered_sentences = [' '.join(sentence) for sentence in filtered_words]
        bag_of_words = FreqDist([word for sentence in filtered_words for word in sentence])
        pos_tags = dict(pos_tag(bag_of_words.keys()))
        tf = {word: count/len(bag_of_words) for word, count in bag_of_words.items()}
        return render_template('index.html', text=text, tokenized_words=tokenized_words, filtered_words=filtered_words, bag_of_words=bag_of_words, pos_tags=pos_tags, sentence_count=sentence_count, tf=tf,sentences=sentences,filtered_sentences=filtered_sentences)

    return render_template('index.html')

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    app.run(debug=True)













