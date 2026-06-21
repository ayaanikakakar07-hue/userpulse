from flask import Flask, render_template, request
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from collections import Counter
import os

# Download NLTK data at startup
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)

app = Flask(__name__)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_filter(text):
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens
            if w not in stop_words and len(w) > 2]


def analyze_review(text):
    cleaned = clean_text(text)
    tokens = tokenize_and_filter(cleaned)
    score = TextBlob(cleaned).sentiment.polarity
    if score > 0.1:
        label = 'Positive'
    elif score < -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    preview = text[:120] + ('...' if len(text) > 120 else '')
    return {'preview': preview, 'score': round(score, 3), 'label': label, 'tokens': tokens}


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    raw = request.form.get('reviews', '')
    reviews = [r.strip() for r in raw.split('\n') if r.strip()]

    if not reviews:
        return render_template('index.html', error='Please paste at least one review.')

    results = [analyze_review(r) for r in reviews]

    counts = Counter(r['label'] for r in results)
    all_tokens = [t for r in results for t in r['tokens']]
    top_keywords = Counter(all_tokens).most_common(12)
    avg_score = round(sum(r['score'] for r in results) / len(results), 3)

    pos_pct = round(counts.get('Positive', 0) / len(results) * 100)
    neg_pct = round(counts.get('Negative', 0) / len(results) * 100)
    neu_pct = round(counts.get('Neutral', 0) / len(results) * 100)

    return render_template('index.html',
                           results=results,
                           counts=counts,
                           top_keywords=top_keywords,
                           avg_score=avg_score,
                           total=len(results),
                           pos_pct=pos_pct,
                           neg_pct=neg_pct,
                           neu_pct=neu_pct,
                           raw=raw)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
