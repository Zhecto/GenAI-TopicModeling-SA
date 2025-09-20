import requests
import pandas as pd
import praw
from datetime import datetime

# ========================
# CONFIG
# ========================
NEWSAPI_KEY = "4ba6c109aca749ef9d2fba6b60bb0a5f"

reddit = praw.Reddit(
    client_id="s03ue3ekn5cHhzpqbIOzaQ",
    client_secret="FHSaYn-k5aVbbIJkWUKolmDqYcZ5FA",
    user_agent="genai-edu-scraper/0.1 by EducationImaginary25"
)

OUTPUT_FILE = "genai_education2.csv"

# ========================
# KEYWORDS
# ========================
EDU_KEYWORDS = [
    "education", "learning", "school", "university",
    "classroom", "student", "teacher", "curriculum", "edtech", "academic"
]

# ========================
# HELPERS
# ========================
def is_genai_edu(text):
    """Check if text explicitly mentions 'generative ai' and has an education context"""
    if not text:
        return False
    low = text.lower()
    return "generative ai" in low and any(e in low for e in EDU_KEYWORDS)

def safe_date(datestr):
    try:
        return datetime.fromisoformat(datestr.replace("Z", "+00:00"))
    except Exception:
        return None

def dedupe(records):
    seen = set()
    out = []
    for r in records:
        key = r.get("url") or r.get("title")
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

# ========================
# SOURCES
# ========================
def fetch_newsapi(query="generative ai education"):
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": "en", "pageSize": 100, "page": 1, "apiKey": NEWSAPI_KEY}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print("NewsAPI error", r.status_code, r.text)
        return []
    data = r.json()
    out = []
    for a in data.get("articles", []):
        title = a.get("title") or ""
        desc = a.get("description") or ""
        body = a.get("content") or ""
        content = " ".join([title, desc, body]).strip()
        if is_genai_edu(content):
            out.append({
                "title": title,
                "content": content,
                "published_at": safe_date(a.get("publishedAt","")),
                "url": a.get("url"),
                "source": a.get("source",{}).get("name","newsapi")
            })
    return out

def fetch_medium():
    url = "https://api.rss2json.com/v1/api.json"
    params = {"rss_url": "https://medium.com/feed/tag/generative-ai"}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print("Medium fetch error", r.status_code)
        return []
    data = r.json()
    out = []
    for item in data.get("items", []):
        title = item.get("title", "")
        content = item.get("content", "")
        text = f"{title} {content}"
        if is_genai_edu(text):
            out.append({
                "title": title,
                "content": text,
                "published_at": safe_date(item.get("pubDate","")),
                "url": item.get("link"),
                "source": "medium"
            })
    return out

def fetch_reddit():
    out = []
    subreddits = "edtech+education+technology+ArtificialIntelligence"

    # Search submissions
    for submission in reddit.subreddit(subreddits).search("generative ai education", limit=50):
        text = f"{submission.title} {submission.selftext}"
        if is_genai_edu(text):
            out.append({
                "title": submission.title,
                "content": text,
                "published_at": datetime.utcfromtimestamp(submission.created_utc),
                "url": f"https://www.reddit.com{submission.permalink}",
                "source": "reddit_post"
            })

        # Fetch comments
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            ctext = comment.body
            if is_genai_edu(ctext):
                out.append({
                    "title": f"Comment on: {submission.title}",
                    "content": ctext,
                    "published_at": datetime.utcfromtimestamp(comment.created_utc),
                    "url": f"https://www.reddit.com{comment.permalink}",
                    "source": "reddit_comment"
                })

    return out

# ========================
# MAIN
# ========================
def main():
    collected = []
    print("Fetching NewsAPI...")
    collected.extend(fetch_newsapi())
    print("Fetching Medium...")
    collected.extend(fetch_medium())
    print("Fetching Reddit...")
    collected.extend(fetch_reddit())

    print(f"Collected {len(collected)} raw items")
    final = dedupe([r for r in collected if is_genai_edu(r.get("content",""))])
    print(f"Filtered down to {len(final)} Generative AI in Education articles")

    if final:
        df = pd.DataFrame(final)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(final)} articles to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
