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

# Pain point theme taxonomy
PAIN_THEMES = {
    'Crashes & Bugs':      ['crash', 'bug', 'freeze', 'error', 'broken', 'glitch', 'fix', 'issue', 'problem', 'malfunction'],
    'Performance':         ['slow', 'lag', 'load', 'loading', 'battery', 'drain', 'speed', 'fast', 'quick', 'delay'],
    'Ads & Monetisation':  ['ad', 'ads', 'advertisement', 'commercial', 'popup', 'banner', 'paid', 'subscription', 'money'],
    'Algorithm & Content': ['algorithm', 'fyp', 'feed', 'recommend', 'content', 'video', 'trending', 'discover', 'show', 'watch'],
    'UI & Design':         ['interface', 'ui', 'design', 'layout', 'button', 'navigation', 'update', 'new', 'old', 'look'],
    'Notifications':       ['notification', 'notify', 'alert', 'push', 'ping', 'spam', 'message', 'reminder'],
    'Account & Trust':     ['ban', 'banned', 'shadow', 'account', 'login', 'privacy', 'data', 'report', 'support', 'unfair'],
    'Upload & Quality':    ['upload', 'quality', 'blurry', 'resolution', 'audio', 'sound', 'video', 'export', 'compress'],
}

FEATURE_RECS = {
    'Crashes & Bugs':      ('P0 — Stability', 'Fix crash loops and silent failures; add crash reporting with auto-retry.'),
    'Performance':         ('P0 — Performance', 'Reduce battery drain; optimize cold-start and video load times.'),
    'Ads & Monetisation':  ('P1 — Ad Experience', 'Cap ad frequency; add rewarded-ad opt-in; improve relevance targeting.'),
    'Algorithm & Content': ('P1 — Discovery', 'Expose user controls for FYP tuning; improve cold-start for new users.'),
    'UI & Design':         ('P2 — UX Polish', 'Audit post-update regressions; run usability tests before major rollouts.'),
    'Notifications':       ('P2 — Notification UX', 'Add granular notification controls; respect OS-level "focus" modes.'),
    'Account & Trust':     ('P1 — Trust & Safety', 'Improve shadow-ban transparency; add appeals flow; faster support SLA.'),
    'Upload & Quality':    ('P2 — Creator Tools', 'Preserve upload resolution; improve in-app editing quality pipeline.'),
}

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

    /* PM REPORT */
    .report-card{background:#fff;border-radius:14px;border:1px solid #e8eaf0;box-shadow:0 2px 12px rgba(0,0,0,.05);padding:1.75rem;max-width:760px;margin:0 auto 1.5rem}
    .report-header{display:flex;align-items:center;gap:.6rem;margin-bottom:1.2rem}
    .report-header h2{font-size:1rem;font-weight:700;color:#1a1a2e}
    .report-chip{background:#6c63ff;color:#fff;border-radius:99px;padding:.2rem .7rem;font-size:.72rem;font-weight:600;letter-spacing:.3px}
    .theme-list{display:flex;flex-direction:column;gap:.75rem}
    .theme-row{border-radius:10px;border:1px solid #e8eaf0;padding:.9rem 1rem;display:flex;gap:1rem;align-items:flex-start}
    .theme-row.pain{border-left:3px solid #e53935}
    .theme-row.positive{border-left:3px solid #43a047}
    .theme-count{font-size:1.4rem;font-weight:800;color:#6c63ff;min-width:2rem;text-align:center;line-height:1}
    .theme-info{flex:1}
    .theme-name{font-size:.88rem;font-weight:700;color:#1a1a2e}
    .theme-quotes{margin-top:.3rem;font-size:.78rem;color:#666;font-style:italic;line-height:1.45}
    .rec-list{display:flex;flex-direction:column;gap:.75rem;margin-top:.25rem}
    .rec-row{border-radius:10px;background:#fafafe;border:1px solid #e8eaf0;padding:.85rem 1rem;display:flex;gap:.9rem;align-items:flex-start}
    .pri-badge{flex-shrink:0;padding:.2rem .55rem;border-radius:6px;font-size:.68rem;font-weight:700;letter-spacing:.3px}
    .p0{background:#ffebee;color:#c62828}.p1{background:#fff3e0;color:#e65100}.p2{background:#e8f5e9;color:#2e7d32}
    .rec-title{font-size:.85rem;font-weight:700;color:#1a1a2e}
    .rec-desc{font-size:.78rem;color:#666;margin-top:.2rem;line-height:1.45}
    .divider{border:none;border-top:1px solid #e8eaf0;margin:1.2rem 0}
    .exec-summary{background:#f0eeff;border-radius:10px;padding:.9rem 1rem;font-size:.87rem;color:#2d2b55;line-height:1.55;margin-bottom:1.2rem}
    .exec-summary strong{color:#4a3fbf}
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

  <!-- PM REPORT -->
  <div class="report-card">
    <div class="report-header">
      <h2>PM Insight Report</h2>
      <span class="report-chip">Auto-Generated</span>
    </div>
    <div class="exec-summary">{{ exec_summary | safe }}</div>

    {% if pain_themes %}
    <div class="section-title">Top Pain Points</div>
    <div class="theme-list" style="margin-bottom:1.2rem">
      {% for theme, count, quotes in pain_themes %}
      <div class="theme-row pain">
        <div class="theme-count">{{ count }}</div>
        <div class="theme-info">
          <div class="theme-name">{{ theme }}</div>
          <div class="theme-quotes">{{ quotes }}</div>
        </div>
      </div>
      {% endfor %}
    </div>
    {% endif %}

    {% if pos_themes %}
    <div class="section-title">What Users Love</div>
    <div class="theme-list" style="margin-bottom:1.2rem">
      {% for theme, count, quotes in pos_themes %}
      <div class="theme-row positive">
        <div class="theme-count">{{ count }}</div>
        <div class="theme-info">
          <div class="theme-name">{{ theme }}</div>
          <div class="theme-quotes">{{ quotes }}</div>
        </div>
      </div>
      {% endfor %}
    </div>
    {% endif %}

    <hr class="divider">
    <div class="section-title">Feature Recommendations</div>
    <div class="rec-list">
      {% for pri_label, pri_class, title, desc in recommendations %}
      <div class="rec-row">
        <span class="pri-badge {{ pri_class }}">{{ pri_label }}</span>
        <div>
          <div class="rec-title">{{ title }}</div>
          <div class="rec-desc">{{ desc }}</div>
        </div>
      </div>
      {% endfor %}
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
    return {'preview': text[:120] + ('...' if len(text) > 120 else ''), 'score': round(score, 3), 'label': label, 'tokens': tokens, 'raw': text}


def classify_themes(results):
    """Map each review to pain themes and positive themes based on keyword overlap."""
    pain_hits = Counter()   # theme -> count
    pos_hits  = Counter()
    pain_examples = {}      # theme -> [quote snippets]
    pos_examples  = {}

    for r in results:
        tokens = set(r['tokens'])
        raw_lower = r['raw'].lower()
        raw_words = set(re.split(r'\W+', raw_lower))
        for theme, keywords in PAIN_THEMES.items():
            if any(k in tokens or k in raw_words for k in keywords):
                if r['label'] == 'Negative':
                    pain_hits[theme] += 1
                    pain_examples.setdefault(theme, []).append(r['raw'][:70])
                elif r['label'] == 'Positive':
                    pos_hits[theme] += 1
                    pos_examples.setdefault(theme, []).append(r['raw'][:70])

    pain_themes = [(t, c, '" ' + pain_examples[t][0] + '…"') for t, c in pain_hits.most_common(4) if c > 0]
    pos_themes  = [(t, c, '" ' + pos_examples[t][0]  + '…"') for t, c in pos_hits.most_common(3)  if c > 0]
    return pain_themes, pos_themes


def build_recommendations(pain_themes, pos_themes):
    seen = set()
    recs = []
    for theme, _, _ in pain_themes:
        if theme in FEATURE_RECS and theme not in seen:
            label, desc = FEATURE_RECS[theme]
            pri = label.split(' — ')[0]
            title = label.split(' — ')[1]
            css = {'P0': 'p0', 'P1': 'p1', 'P2': 'p2'}.get(pri, 'p2')
            recs.append((pri, css, title, desc))
            seen.add(theme)
    # sort P0 first
    recs.sort(key=lambda x: x[0])
    return recs


def build_exec_summary(results, counts, pain_themes, pos_themes, avg_score):
    total = len(results)
    neg_n = counts.get('Negative', 0)
    pos_n = counts.get('Positive', 0)
    sentiment_word = 'predominantly negative' if avg_score < -0.05 else ('mixed' if avg_score < 0.15 else 'broadly positive')
    top_pain = pain_themes[0][0] if pain_themes else 'general usability'
    top_pos  = pos_themes[0][0]  if pos_themes  else 'core experience'
    return (
        f"Across {total} reviews, sentiment is {sentiment_word} (avg polarity {avg_score:+.2f}). "
        f"<strong>{neg_n} negative reviews</strong> cluster around <strong>{top_pain}</strong> — the highest-signal pain area. "
        f"<strong>{pos_n} positive reviews</strong> highlight <strong>{top_pos}</strong> as the product's strongest asset. "
        f"Immediate attention should focus on P0 stability and performance issues before investing further in growth features."
    )


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

    pain_themes, pos_themes = classify_themes(results)
    recommendations = build_recommendations(pain_themes, pos_themes)
    exec_summary = build_exec_summary(results, counts, pain_themes, pos_themes, avg_score)

    return render_template_string(HTML, results=results, counts=counts, top_keywords=top_keywords,
                                  avg_score=avg_score, total=len(results),
                                  pos_pct=pos_pct, neg_pct=neg_pct, neu_pct=neu_pct, raw=raw,
                                  pain_themes=pain_themes, pos_themes=pos_themes,
                                  recommendations=recommendations, exec_summary=exec_summary)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
