import re, pandas as pd, os, time as tm, random, nltk
from datetime import datetime
from playwright.sync_api import sync_playwright
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter

class TwitterScraper:
    def __init__(self):
        self.tweets_data = []
        self.hashtags = ["naukri", "jobs", "jobseeker", "vacancy"]
        self.target_count, self.scroll_attempts, self.max_scroll_attempts = 2000, 0, 300
        self.seen_tweets, self.user_data_dir = set(), "twitter_session"
        nltk.download('stopwords', quiet=True), nltk.download('punkt', quiet=True)
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', str(text).lower(), flags=re.MULTILINE)
        return ' '.join([w for w in nltk.word_tokenize(re.sub(r'[^a-zA-Z0-9\s@#]', '', text)) 
                        if w not in self.stop_words and len(w) > 2])

    def scrape_tweets(self):
        with sync_playwright() as pw:
            browser = pw.chromium.launch_persistent_context(
                self.user_data_dir, headless=False, channel="chrome",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                viewport={"width": 1280, "height": 720}, timeout=120000)
            page = browser.new_page()
            self.handle_login(page)
            [self.scrape_hashtag(page, h) for h in self.hashtags if len(self.tweets_data) < self.target_count]
            self.save_data(), self.analyze_data(), browser.close()

    def handle_login(self, page):
        page.goto("https://twitter.com/login", timeout=60000)
        try: page.wait_for_selector("text=Home", timeout=15000)
        except: print("Manual login required"); page.wait_for_selector("text=Home", timeout=120000)

    def scrape_hashtag(self, page, hashtag):
        print(f"\nScraping #{hashtag}...")
        for attempt in range(3):
            try:
                page.goto(f"https://twitter.com/search?q=%23{hashtag}&src=typed_query&f=live", timeout=60000)
                page.wait_for_selector('article[data-testid="tweet"]', timeout=30000)
                break
            except Exception as e:
                if attempt == 2: return print(f"Failed to load #{hashtag}")
                tm.sleep(5)
        
        last_pos, no_new = page.evaluate("window.scrollY"), 0
        while len(self.tweets_data) < self.target_count and self.scroll_attempts < self.max_scroll_attempts:
            self.scroll_attempts += 1
            page.keyboard.press("End"), tm.sleep(random.uniform(1.5, 3.5))
            self.dismiss_popups(page)
            tweets = page.query_selector_all('article[data-testid="tweet"]')
            new_tweets = self.process_tweets(tweets[-25:])
            curr_pos = page.evaluate("window.scrollY")
            no_new = no_new + 1 if curr_pos == last_pos else 0
            if no_new > 5: break
            last_pos, no_new = (curr_pos, 0) if curr_pos != last_pos else (last_pos, no_new)
            print(f"Progress: {len(self.tweets_data)}/{self.target_count} tweets")

    def dismiss_popups(self, page):
        for text in ["Sign up", "Log in"]:
            if page.query_selector(f"text='{text}'"): page.mouse.click(10, 10), tm.sleep(1)

    def process_tweets(self, tweets):
        new = 0
        for t in tweets:
            try:
                c = t.query_selector("div[data-testid='tweetText']").inner_text()
                if c in self.seen_tweets: continue
                un = t.query_selector("div[data-testid='User-Name']").inner_text().strip()
                dt = t.query_selector("time").get_attribute("datetime").split('T')
                self.tweets_data.append({
                    "Username": un, "Tweet": c.strip(), "Cleaned_Tweet": self.preprocess_text(c),
                    "Date": dt[0], "Time": dt[1].split('.')[0],
                    "Mentions": ', '.join(set(re.findall(r'@(\w+)', c))),
                    "Hashtags": ', '.join(set(re.findall(r'#(\w+)', c))),
                    "Likes": self.get_metric(t, "div[data-testid='like']"),
                    "Retweets": self.get_metric(t, "div[data-testid='retweet']"),
                    "Comments": self.get_metric(t, "div[data-testid='reply']"),
                    "Views": self.get_metric(t, "div[aria-label*='views']", True),
                    "Replies": self.estimate_replies(t),
                    "Sentiment": TextBlob(c).sentiment.polarity
                })
                self.seen_tweets.add(c)
                new += 1
            except: continue
        return new

    def get_metric(self, el, sel, is_aria=False):
        try:
            m = el.query_selector(sel)
            if not m: return "0"
            return re.search(r'\d+', m.get_attribute("aria-label").replace(',', ''))[0] if is_aria else m.inner_text().replace(',', '') or "0"
        except: return "0"

    def estimate_replies(self, t):
        try:
            r = t.query_selector("div[data-testid='reply']")
            return re.search(r'\d+', r.get_attribute("aria-label").replace(',', ''))[0] if r and "replies" in r.get_attribute("aria-label").lower() else "0"
        except: return "0"

    def save_data(self):
        if not self.tweets_data: return print("No data to save")
        df = pd.DataFrame(self.tweets_data)
        os.makedirs("twitter_data", exist_ok=True)
        cols = ['Likes', 'Retweets', 'Comments', 'Replies', 'Views']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        if os.path.exists(fn := "twitter_data/tweets_combined.csv"):
            existing = pd.read_csv(fn)
            pd.concat([existing, df]).drop_duplicates(subset=['Tweet']).to_csv(fn, index=False)
            print(f"\nAppended {len(df)} tweets. Total: {len(existing) + len(df)}")
        else: df.to_csv(fn, index=False), print(f"\nSaved {len(df)} tweets")

    def analyze_data(self):
        if not self.tweets_data: return print("No data to analyze")
        df = pd.DataFrame(self.tweets_data)
        df['Sentiment_Label'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral')
        all_tags = [tag for sub in df['Hashtags'].str.split(', ') for tag in sub if tag]
        stats = df[['Likes', 'Retweets', 'Comments', 'Replies', 'Views']].describe()
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        hourly = df.groupby(df['DateTime'].dt.hour).size()
        
        with open("twitter_data/analysis_report.txt", "w") as f:
            f.write(f"""Twitter Data Analysis Report
===========================
Collected Tweets: {len(df)}
Time Period: {df['Date'].min()} to {df['Date'].max()}

Key Metrics:
- Average Likes: {stats.loc['mean', 'Likes']:.1f}
- Average Retweets: {stats.loc['mean', 'Retweets']:.1f}
- Average Views: {stats.loc['mean', 'Views']:.1f}

Sentiment Analysis:
- Positive: {(df['Sentiment_Label'] == 'Positive').sum()}
- Neutral: {(df['Sentiment_Label'] == 'Neutral').sum()}
- Negative: {(df['Sentiment_Label'] == 'Negative').sum()}

Top 10 Hashtags:
{chr(10).join([f'{t}: {c}' for t,c in Counter(all_tags).most_common(10)])}

Engagement Patterns:
- Peak: {hourly.idxmax()}:00 ({hourly.max()} tweets)
- Lowest: {hourly.idxmin()}:00 ({hourly.min()} tweets)""")

        plt.figure(figsize=(15, 10))
        plt.subplot(2,2,1), pd.Series(dict(Counter(all_tags).most_common(10))).plot(kind='bar'), plt.title('Top 10 Hashtags')
        plt.subplot(2,2,2), df['Sentiment_Label'].value_counts().plot(kind='pie', autopct='%1.1f%%'), plt.title('Sentiment')
        plt.subplot(2,2,3), hourly.plot(kind='line'), plt.title('Hourly Activity'), plt.xlabel('Hour'), plt.ylabel('Tweets')
        plt.tight_layout(), plt.savefig("twitter_data/visualizations.png")
        print("Analysis complete")

if __name__ == "__main__":
    TwitterScraper().scrape_tweets()
