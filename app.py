from flask import Flask, render_template_string, request
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from collections import Counter
import os

for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)

app = Flask(__name__)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>UserPulse — App Review Sentiment Analyzer</title>
  <style>
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f7f8fc;color:#1a1a2e;min-height:100vh}
    nav{background:#fff;border-bottom:1px solid #e8eaf0;padding:0 2rem;display:flex;align-items:center;height:60px;gap:.5rem}
    .logo-dot{width:10px;height:10px;border-radius:50%;background:#6c63ff}
    .logo-text{font-size:1.15rem;font-weight:700;color:#1a1a2e;letter-spacing:-.3px}
    .logo-sub{font-size:.8rem;color:#888;margin-left:.3rem}
    .hero{text-align:center;padding:3.5rem 1rem 2rem}
    .hero h1{font-size:2rem;font-weight:800;letter-spacing:-.5px}
    .hero h1 span{color:#6c63ff}
    .hero p{margin-top:.6rem;color:#666;font-size:1rem}
    .card{background:#fff;border-radius:14px;border:1px solid #e8eaf0;box-shadow:0 2px 12px rgba(0,0,0,.05);padding:1.75rem;max-width:760px;margin:0 auto 1.5rem}
    .card label{display:block;font-size:.82rem;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.5px;margin-bottom:.6rem}
    textarea{width:100%;height:160px;border:1.5px solid #e0e2ef;border-radius:10px;padding:.9rem 1rem;font-size:.92rem;color:#1a1a2e;resize:vertical;outline:none;transition:border-color .2s;font-family:inherit;line-height:1.5}
    textarea:focus{border-color:#6c63ff}
    .hint{font-size:.78rem;color:#aaa;margin-top:.4rem}
    .btn{margin-top:1rem;background:#6c63ff;color:#fff;border:none;padding:.75rem 2rem;border-radius:8px;font-size:.95rem;font-weight:600;cursor:pointer;transition:background .2s}
    .btn:hover{background:#574fd6}
    .error{color:#e53935;font-size:.88rem;margin-top:.5rem}
    .stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;max-width:760px;margin:0 auto 1.5rem}
    .stat-card{background:#fff;border-radius:12px;border:1px solid #e8eaf0;padding:1.1rem 1rem;text-align:center}
    .stat-card .num{font-size:1.8rem;font-weight:800;line-height:1}
    .stat-card .lbl{font-size:.75rem;color:#888;margin-top:.3rem;text-transform:uppercase;letter-spacing:.4px}
    .positive .num{color:#2e7d32}.negative .num{color:#c62828}.neutral .num{color:#e65100}.total .num{color:#6c63ff}
    .bar-wrap{max-width:760px;margin:0 auto 1.5rem}
    .bar-track{height:10px;border-radius:99px;background:#f0f0f5;overflow:hidden;display:flex}
    .bar-pos{background:#43a047;height:100%}.bar-neg{background:#e53935;height:100%}.bar-neu{background:#fb8c00;height:100%}
    .bar-legend{display:flex;gap:1.2rem;margin-top:.5rem;font-size:.78rem;color:#666}
    .bar-legend span{display:flex;align-items:center;gap:.3rem}
    .dot{width:8px;height:8px;border-radius:50%;display:inline-block}
    .dot-pos{background:#43a047}.dot-neg{background:#e53935}.dot-neu{background:#fb8c00}
    .keywords{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.75rem}
    .kw{background:#f0eeff;color:#4a3fbf;border-radius:99px;padding:.3rem .75rem;font-size:.8rem;font-weight:500}
    .results-list{max-width:760px;margin:0 auto}
    .result-item{background:#fff;border:1px solid #e8eaf0;border-radius:10px;padding:1rem 1.1rem;margin-bottom:.75rem;display:flex;align-items:flex-start;gap:1rem}
    .badge{flex-shrink:0;padding:.25rem .65rem;border-radius:99px;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.4px}
    .badge-Positive{background:#e8f5e9;color:#2e7d32}.badge-Negative{background:#ffebee;color:#c62828}.badge-Neutral{background:#fff3e0;color:#e65100}
    .result-text{font-size:.88rem;color:#444;line-height:1.5;flex:1}
    .result-score{font-size:.78rem;color:#aaa;margin-top:.3rem}
    .section-title{font-size:.8rem;font-weight:700;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-bottom:.75rem}
    .wrapper{padding:0 1rem 3rem}
  </style>
</head>
<body>
<nav>
  <div class="logo-dot"></div>
  <span class="logo-text">UserPulse</span>
  <span class="logo-sub">App Review Sentiment Analyzer</span>
</nav>
<div class="hero">
  <h1>Understand what your users <span>really</span> think</h1>
  <p>Paste app reviews below — UserPulse scores sentiment, surfaces pain points, and delivers PM-ready insights.</p>
</div>
<div class="wrapper">
  <div class="card">
    <form method="POST" action="/analyze">
      <label for="reviews">Paste Reviews (one per line)</label>
      <textarea id="reviews" name="reviews" placeholder="This app crashes every time I open a video...&#10;Love the algorithm, always shows what I want&#10;Too many ads, getting annoying">{{ raw or "" }}</textarea>
      <div class="hint">Try pasting 5–20 reviews for a meaningful breakdown.</div>
      {% if error %}<div class="error">{{ error }}</div>{% endif %}
      <button class="btn" type="submit">Analyze Reviews</button>
    </form>
  </div>
  {% if results %}
  <div class="stats-grid">
    <div class="stat-card total"><div class="num">{{ total }}</div><div class="lbl">Reviews</div></div>
    <div class="stat-card positive"><div class="num">{{ pos_pct }}%</div><div class="lbl">Positive</div></div>
    <div class="stat-card negative"><div class="num">{{ neg_pct }}%</div><div class="lbl">Negative</div></div>
    <div class="stat-card neutral"><div class="num">{{ avg_score }}</div><div class="lbl">Avg Score</div></div>
  </div>
  <div class="bar-wrap">
    <div class="bar-track">
      <div class="bar-pos" style="width:{{ pos_pct }}%"></div>
      <div class="bar-neu" style="width:{{ neu_pct }}%"></div>
      <div class="bar-neg" style="width:{{ neg_pct }}%"></div>
    </div>
    <div class="bar-legend">
      <span><i class="dot dot-pos"></i> {{ counts.get("Positive", 0) }} Positive</span>
      <span><i class="dot dot-neu"></i> {{ counts.get("Neutral", 0) }} Neutral</span>
      <span><i class="dot dot-neg"></i> {{ counts.get("Negative", 0) }} Negative</span>
    </div>
  </div>
  <div class="card">
    <div class="section-title">Top Keywords</div>
    <div class="keywords">
      {% for word, count in top_keywords %}
      <span class="kw">{{ word }} <strong>{{ count }}</strong></span>
      {% endfor %}
    </div>
  </div>
  <div class="results-list">
    <div class="section-title">Review Breakdown</div>
    {% for r in results %}
    <div class="result-item">
      <span class="badge badge-{{ r.label }}">{{ r.label }}</span>
      <div>
        <div class="result-text">{{ r.preview }}</div>
        <div class="result-score">Polarity score: {{ r.score }}</div>
      </div>
    </div>
    {% endfor %}
  </div>
  {% endif %}
</div>
</body>
</html>'''


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize_and_filter(text):
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]


def analyze_review(text):
    cleaned = clean_text(text)
    tokens = tokenize_and_filter(cleaned)
    score = TextBlob(cleaned).sentiment.polarity
    label = 'Positive' if score > 0.1 else ('Negative' if score < -0.05 else 'Neutral')
    return {'preview': text[:120] + ('...' if len(text) > 120 else ''), 'score': round(score, 3), 'label': label, 'tokens': tokens}


@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML)


@app.route('/analyze', methods=['POST'])
def analyze():
    raw = request.form.get('reviews', '')
    reviews = [r.strip() for r in raw.split('\n') if r.strip()]
    if not reviews:
        return render_template_string(HTML, error='Please paste at least one review.')
    results = [analyze_review(r) for r in reviews]
    counts = Counter(r['label'] for r in results)
    all_tokens = [t for r in results for t in r['tokens']]
    top_keywords = Counter(all_tokens).most_common(12)
    avg_score = round(sum(r['score'] for r in results) / len(results), 3)
    pos_pct = round(counts.get('Positive', 0) / len(results) * 100)
    neg_pct = round(counts.get('Negative', 0) / len(results) * 100)
    neu_pct = round(counts.get('Neutral', 0) / len(results) * 100)
    return render_template_string(HTML, results=results, counts=counts, top_keywords=top_keywords,
                                  avg_score=avg_score, total=len(results),
                                  pos_pct=pos_pct, neg_pct=neg_pct, neu_pct=neu_pct, raw=raw)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
