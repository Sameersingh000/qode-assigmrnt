import re, pandas as pd, os, time, random, nltk
from datetime import datetime
from playwright.sync_api import sync_playwright
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter

class TwitterScraper:
    def __init__(self):
        nltk.download(['stopwords', 'punkt'], quiet=True)
        self.tweets_data = []
        self.hashtags = ["naukri", "jobs", "jobseeker", "vacancy"]
        self.target_count, self.scroll_attempts, self.max_scroll_attempts = 2000, 0, 300
        self.seen_tweets, self.user_data_dir = set(), "twitter_session"
        self.stop_words, self.hashtag_limits = set(stopwords.words('english')), {h:500 for h in self.hashtags}
        self.current_hashtag_index, self.max_attempts_per_hashtag = 0, 3
        self.min_tweets_per_round, self.session_timeout = 50, 120
        self.rate_limit_wait, self.rate_limit_retries = 600, 3

    def preprocess_text(self, text):
        text = re.sub(r'http\S+|www\S+|https\S+', '', str(text).lower(), flags=re.MULTILINE)
        return ' '.join([w for w in nltk.word_tokenize(re.sub(r'[^a-zA-Z0-9\s@#]', '', text)) 
                        if w not in self.stop_words and len(w) > 2])

    def scrape_tweets(self):
        with sync_playwright() as pw:
            browser = self.init_browser(pw)
            page = browser.new_page()
            self.handle_login(page)
            
            start_time, rate_limit_retries = time.time(), 0
            while (len(self.tweets_data) < self.target_count and self.scroll_attempts < self.max_scroll_attempts 
                   and time.time() - start_time < 3600 and rate_limit_retries < self.rate_limit_retries):
                try:
                    for i, hashtag in enumerate(self.hashtags):
                        if self.hashtag_limits[hashtag] <= 0: continue
                        print(f"\nTargets: Total {len(self.tweets_data)}/{self.target_count} | #{hashtag} {500-self.hashtag_limits[hashtag]}/500")
                        
                        if self.scrape_hashtag(page, hashtag) == "rate_limit":
                            rate_limit_retries += 1
                            time.sleep(self.rate_limit_wait)
                            page.close(); browser.close()
                            browser, page = self.init_browser(pw), browser.new_page()
                            self.handle_login(page)
                            i -= 1
                            continue
                            
                        if len(self.tweets_data) >= self.target_count: break
                        time.sleep(random.uniform(2, 5))
                except Exception as e: print(f"Error: {str(e)}")
            
            self.save_data(); self.analyze_data(); browser.close()

    def init_browser(self, pw):
        return pw.chromium.launch_persistent_context(
            self.user_data_dir, headless=False, channel="chrome",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            viewport={"width": 1280, "height": 720}, timeout=120000)

    def handle_login(self, page):
        page.goto("https://twitter.com/login", timeout=60000)
        try: page.wait_for_selector("text=Home", timeout=15000)
        except: 
            print("Manual login required")
            page.wait_for_selector("text=Home", timeout=120000)

    def scrape_hashtag(self, page, hashtag):
        attempts, collected_in_session, session_start = 0, 0, time.time()
        while (self.hashtag_limits[hashtag] > 0 and attempts < self.max_attempts_per_hashtag 
               and len(self.tweets_data) < self.target_count and time.time() - session_start < self.session_timeout):
            try:
                if attempts > 0 or collected_in_session == 0:
                    page.goto(f"https://twitter.com/search?q=%23{hashtag}&src=typed_query&f=live", timeout=60000)
                    try: page.wait_for_selector('article[data-testid="tweet"]', timeout=30000)
                    except: return "rate_limit" if page.query_selector("text=Rate limit exceeded") else ""
                    time.sleep(random.uniform(2, 4))
                
                last_pos, no_new_count, session_tweets = page.evaluate("window.scrollY"), 0, 0
                while (self.hashtag_limits[hashtag] > 0 and session_tweets < self.min_tweets_per_round 
                       and no_new_count < 5 and len(self.tweets_data) < self.target_count 
                       and time.time() - session_start < self.session_timeout):
                    self.scroll_attempts += 1
                    page.keyboard.press("End"); time.sleep(random.uniform(1.5, 3.5))
                    self.dismiss_popups(page)
                    if page.query_selector("text=Rate limit exceeded"): return "rate_limit"
                    
                    tweets = page.query_selector_all('article[data-testid="tweet"]')
                    new_tweets = self.process_tweets(tweets[-20:], hashtag)
                    session_tweets += new_tweets; collected_in_session += new_tweets
                    
                    curr_pos = page.evaluate("window.scrollY")
                    no_new_count = no_new_count + 1 if curr_pos == last_pos else 0
                    last_pos = curr_pos
                    print(f"Collected: {new_tweets} new | Total: {len(self.tweets_data)}/{self.target_count} | #{hashtag}: {500-self.hashtag_limits[hashtag]}/500")
                    
                    if no_new_count > 3: print("No new tweets, moving on..."); break
                
                if session_tweets < 10: 
                    attempts += 1; print(f"Attempt {attempts}/{self.max_attempts_per_hashtag} for #{hashtag}")
                    time.sleep(5)
            except Exception as e: 
                print(f"Error scraping #{hashtag}: {str(e)}"); attempts += 1; time.sleep(5)

    def dismiss_popups(self, page):
        if any(page.query_selector(f"text='{t}'") for t in ["Sign up", "Log in"]): 
            page.mouse.click(10, 10); time.sleep(1)

    def process_tweets(self, tweets, hashtag):
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
                    "Sentiment": TextBlob(c).sentiment.polarity,
                    "Source_Hashtag": hashtag
                })
                
                self.seen_tweets.add(c); new += 1; self.hashtag_limits[hashtag] -= 1
                if self.hashtag_limits[hashtag] <= 0: break
            except: continue
        return new

    def get_metric(self, el, sel, is_aria=False):
        try:
            m = el.query_selector(sel)
            if not m: return "0"
            text = m.get_attribute("aria-label") if is_aria else m.inner_text()
            return re.search(r'\d+', text.replace(',', ''))[0] if text else "0"
        except: return "0"

    def estimate_replies(self, t):
        try:
            r = t.query_selector("div[data-testid='reply']")
            return re.search(r'\d+', r.get_attribute("aria-label").replace(',', ''))[0] if r and "replies" in r.get_attribute("aria-label").lower() else "0"
        except: return "0"

    def save_data(self):
        if not self.tweets_data: return print("No data to save")
        os.makedirs("twitter_data", exist_ok=True)
        df = pd.DataFrame(self.tweets_data)
        df[['Likes', 'Retweets', 'Comments', 'Replies', 'Views']] = df[['Likes', 'Retweets', 'Comments', 'Replies', 'Views']].apply(pd.to_numeric, errors='coerce').fillna(0)
        filename = "twitter_data/tweets_combined.csv"
        if os.path.exists(filename):
            df = pd.concat([pd.read_csv(filename), df]).drop_duplicates(subset=['Tweet'])
        df.to_csv(filename, index=False)
        print(f"\nSaved {len(df)} tweets to {filename}")

    def analyze_data(self):
        if not self.tweets_data: return print("No data to analyze")
        df = pd.DataFrame(self.tweets_data)
        df['Sentiment_Label'] = df['Sentiment'].apply(lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral')
        all_tags = [tag for sub in df['Hashtags'].str.split(', ') for tag in sub if tag]
        stats = df[['Likes', 'Retweets', 'Comments', 'Replies', 'Views']].describe()
        hourly = df.groupby(pd.to_datetime(df['Date'] + ' ' + df['Time']).dt.hour.size())
        
        with open("twitter_data/analysis_report.txt", "w") as f:
            f.write(f"""Twitter Data Analysis Report
===========================
Collected Tweets: {len(df)}
Time Period: {df['Date'].min()} to {df['Date'].max()}

Hashtag Distribution:
{chr(10).join([f'#{k}: {v}' for k,v in df['Source_Hashtag'].value_counts().items()])}

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
        plots = [
            ('Top 10 Hashtags', pd.Series(dict(Counter(all_tags).most_common(10)).plot(kind='bar'), None)),
            ('Sentiment Distribution', df['Sentiment_Label'].value_counts().plot(kind='pie', autopct='%1.1f%%'), None),
            ('Hourly Activity', hourly.plot(kind='line'), ('Hour', 'Tweets')),
            ('Tweets per Search Hashtag', df['Source_Hashtag'].value_counts().plot(kind='bar'), None)
        ]
        for i, (title, plot, labels) in enumerate(plots, 1):
            plt.subplot(2, 2, i); plot; plt.title(title)
            if labels: plt.xlabel(labels[0]); plt.ylabel(labels[1])
        plt.tight_layout(); plt.savefig("twitter_data/visualizations.png")
        print("Analysis complete. Report saved to twitter_data/analysis_report.txt")

if __name__ == "__main__":
    TwitterScraper().scrape_tweets()
