import re
import pandas as pd
from playwright.sync_api import sync_playwright
import time
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging
import os
import random
from tenacity import retry, stop_after_attempt, wait_exponential
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_scraper.log'),
        logging.StreamHandler()
    ]
)

# Ensure NLTK data is downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class TwitterScraper:
    def __init__(self):
        self.tweets_data = []
        self.hashtags = ["naukri", "jobs", "jobseeker", "vacancy"]
        self.target_count = 2000  # Minimum target
        self.scraped_count = 0
        self.duplicate_count = 0
        self.timeout = 300000  # 5 minutes in milliseconds
        self.max_scrolls = 500  # Increased maximum scrolls
        self.user_data_dir = os.path.join(os.getcwd(), "playwright_data")
        self.seen_tweets = set()  # For deduplication
        self.start_time = time.time()
        
        # Create results directory if not exists
        os.makedirs('scraped_results', exist_ok=True)

    def preprocess_text(self, text):
        """Clean and preprocess tweet text"""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words('english')]
        return ' '.join(words)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def scrape_hashtag(self, page, hashtag):
        """Scrape tweets for a specific hashtag with retry logic"""
        logging.info(f"Processing #{hashtag}... (Current count: {self.scraped_count}/{self.target_count})")
        
        try:
            page.goto(
                f"https://twitter.com/search?q=%23{hashtag}%20-filter%3Areplies&src=typed_query&f=live",
                wait_until="domcontentloaded",
                timeout=60000
            )
            
            # Check for login state
            if page.is_visible("text='Log in'"):
                logging.info("Please log in manually in the browser window...")
                page.wait_for_selector("text='Home'", timeout=120000)
            
            page.wait_for_selector('article[data-testid="tweet"]:not([data-scraped])', timeout=60000)
            logging.info("Tweets loaded successfully")

            scroll_count = 0
            last_position = 0
            stale_count = 0
            consecutive_empty = 0

            while (self.scraped_count < self.target_count and 
                   scroll_count < self.max_scrolls):
                
                # Mark tweets as scraped to avoid reprocessing
                page.evaluate('''() => {
                    document.querySelectorAll('article[data-testid="tweet"]:not([data-scraped])')
                        .forEach(el => el.setAttribute('data-scraped', 'true'));
                }''')
                
                tweets = page.query_selector_all('article[data-testid="tweet"][data-scraped="true"]')
                current_count = len(tweets)
                
                if current_count > last_position:
                    new_tweets = tweets[last_position:current_count]
                    processed = self.process_new_tweets(new_tweets)
                    
                    if processed == 0:
                        consecutive_empty += 1
                        if consecutive_empty > 3:
                            logging.warning("No new valid tweets detected in 3 consecutive batches")
                            break
                    else:
                        consecutive_empty = 0
                    
                    last_position = current_count
                    stale_count = 0
                else:
                    stale_count += 1
                    if stale_count > 5:
                        logging.warning("No new tweets detected - breaking scroll loop")
                        break

                # Scroll with random delay
                scroll_delay = max(1, min(5, random.gauss(2, 0.5)))
                page.evaluate("window.scrollBy(0, document.body.scrollHeight * 0.75)")
                time.sleep(scroll_delay)
                
                scroll_count += 1
                logging.info(
                    f"#{hashtag} - Scroll {scroll_count}: "
                    f"Collected {self.scraped_count}/{self.target_count} tweets "
                    f"({self.duplicate_count} duplicates)"
                )

                # Save progress every 50 tweets
                if self.scraped_count % 50 == 0 and self.scraped_count > 0:
                    self.save_to_csv(partial=True)

                # Break if we've been scraping this hashtag for too long with little progress
                if time.time() - self.start_time > 3600 and self.scraped_count < 100:
                    logging.warning("Progress too slow - moving to next hashtag")
                    break

        except Exception as e:
            logging.error(f"Error processing #{hashtag}: {str(e)}")
            page.screenshot(path=f"scraped_results/error_{hashtag}_{int(time.time())}.png")
            raise  # This will trigger retry

    def process_new_tweets(self, tweets):
        """Process and store new tweets"""
        processed_count = 0
        for tweet in tweets:
            try:
                # Extract username
                username_element = tweet.query_selector("div[data-testid='User-Name']")
                username = username_element.inner_text().strip() if username_element else "Unknown"
                
                # Extract tweet content
                content_element = tweet.query_selector("div[data-testid='tweetText']")
                content = content_element.inner_text().strip() if content_element else ""
                
                # Skip if empty content or duplicate
                content_hash = hash(content)
                if not content or content_hash in self.seen_tweets:
                    self.duplicate_count += 1
                    continue
                
                self.seen_tweets.add(content_hash)
                
                # Extract datetime
                time_element = tweet.query_selector("time")
                if time_element:
                    datetime_str = time_element.get_attribute("datetime")
                    date, time = datetime_str.split('T')[0], datetime_str.split('T')[1].split('.')[0]
                else:
                    date, time = "Unknown", "Unknown"
                
                # Extract metrics
                metrics = {
                    'likes': self.get_metric(tweet, "div[data-testid='like']"),
                    'retweets': self.get_metric(tweet, "div[data-testid='retweet']"),
                    'comments': self.get_metric(tweet, "div[data-testid='reply']"),
                    'views': self.get_metric(tweet, "div[aria-label*='views']", is_aria=True)
                }

                # Extract mentions and hashtags
                mentions = ', '.join(re.findall(r'@\w+', content))
                hashtags = ', '.join(re.findall(r'#\w+', content))

                # Store tweet data
                self.tweets_data.append([
                    username,
                    content,
                    date,
                    time,
                    mentions,
                    hashtags,
                    metrics['likes'],
                    metrics['retweets'],
                    metrics['comments'],
                    "0",  # Replies not directly available
                    metrics['views']
                ])
                
                self.scraped_count += 1
                processed_count += 1

            except Exception as e:
                logging.warning(f"Skipping tweet: {str(e)}")
                continue
                
        return processed_count

    def get_metric(self, element, selector, is_aria=False):
        """Extract metric value from tweet"""
        try:
            metric_element = element.query_selector(selector)
            if not metric_element:
                return "0"
            
            if is_aria:
                aria_label = metric_element.get_attribute("aria-label")
                if aria_label:
                    return aria_label.split()[0].replace(',', '')
                return "0"
            
            text = metric_element.inner_text()
            return text.replace(',', '') if text else "0"
        except:
            return "0"

    def scrape_tweets(self):
        """Main scraping function"""
        with sync_playwright() as pw:
            try:
                # Launch browser with persistent context
                browser = pw.chromium.launch_persistent_context(
                    self.user_data_dir,
                    headless=False,
                    channel="chrome",
                    viewport={"width": 1280, "height": 800},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    timeout=self.timeout,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-infobars"
                    ]
                )
                
                page = browser.new_page()
                page.set_default_timeout(self.timeout)

                # Randomize hashtag order to avoid pattern detection
                random.shuffle(self.hashtags)
                
                for hashtag in self.hashtags:
                    if self.scraped_count >= self.target_count:
                        break

                    try:
                        self.scrape_hashtag(page, hashtag)
                        # Random delay between hashtags
                        time.sleep(random.uniform(5, 15))
                    except Exception as e:
                        logging.error(f"Failed to process #{hashtag} after retries: {str(e)}")
                        continue

                # If we still haven't reached target, try alternative approaches
                if self.scraped_count < self.target_count:
                    logging.warning(f"Only collected {self.scraped_count} tweets. Trying alternative methods...")
                    self.try_alternative_sources(page)

            except Exception as e:
                logging.error(f"Fatal error: {str(e)}")
            finally:
                total_time = (time.time() - self.start_time) / 60
                logging.info(f"Scraping completed. Collected {self.scraped_count} tweets in {total_time:.1f} minutes")
                self.save_to_csv()
                input("Press Enter to close the browser...")
                browser.close()

    def try_alternative_sources(self, page):
        """Try alternative methods if primary scraping fails"""
        alternative_queries = [
            "job openings",
            "hiring now",
            "career opportunities",
            "we are hiring"
        ]
        
        for query in alternative_queries:
            if self.scraped_count >= self.target_count:
                break
                
            try:
                logging.info(f"Trying alternative query: '{query}'")
                page.goto(
                    f"https://twitter.com/search?q={query.replace(' ', '%20')}&src=typed_query&f=live",
                    wait_until="domcontentloaded"
                )
                time.sleep(5)
                self.scrape_hashtag(page, query)
            except Exception as e:
                logging.error(f"Failed alternative query '{query}': {str(e)}")

    def save_to_csv(self, partial=False):
        """Save collected tweets to CSV file"""
        df = pd.DataFrame(self.tweets_data, columns=[
            "Username", "Tweet", "Date", "Time", 
            "Mentions", "Hashtags", "Likes", 
            "Retweets", "Comments", "Replies", "Views"
        ])
        
        # Clean and preprocess tweets
        df['Cleaned_Tweet'] = df['Tweet'].apply(self.preprocess_text)
        
        # Convert metrics to numeric
        for col in ["Likes", "Retweets", "Comments", "Replies", "Views"]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        filename = "scraped_results/tweets_partial.csv" if partial else "scraped_results/tweets_final.csv"
        df.to_csv(filename, index=False)
        
        if partial:
            logging.info(f"Progress saved: {len(df)} tweets (partial)")
        else:
            logging.info(f"Final dataset saved: {len(df)} tweets")

    def analyze_tweets(self):
        """Perform analysis on collected tweets"""
        try:
            from textblob import TextBlob
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            from sklearn.feature_extraction.text import CountVectorizer
        except ImportError:
            logging.error("Analysis libraries not installed. Run: pip install textblob wordcloud matplotlib scikit-learn")
            return
        
        # Load data
        df = pd.read_csv("scraped_results/tweets_final.csv")
        
        # Ensure we have enough data
        if len(df) < 100:
            logging.warning("Insufficient data for meaningful analysis")
            return
        
        # 1. Sentiment Analysis
        df['Sentiment'] = df['Cleaned_Tweet'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        
        # Categorize sentiment
        df['Sentiment_Label'] = pd.cut(
            df['Sentiment'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        # 2. Popular Hashtags Analysis
        all_hashtags = []
        for tags in df['Hashtags'].dropna():
            all_hashtags.extend(tags.split(', '))
            
        hashtag_counts = pd.Series(all_hashtags).value_counts().head(20)
        
        # 3. Word Cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
            ' '.join(df['Cleaned_Tweet'])
        )
        
        # 4. Engagement Analysis
        engagement_stats = df[['Likes', 'Retweets', 'Comments', 'Views']].describe()
        
        # Save analysis results
        analysis_results = {
            'sentiment_distribution': df['Sentiment_Label'].value_counts(normalize=True),
            'top_hashtags': hashtag_counts,
            'engagement_stats': engagement_stats
        }
        
        # Generate visualizations
        plt.figure(figsize=(15, 10))
        
        # Sentiment distribution
        plt.subplot(2, 2, 1)
        df['Sentiment_Label'].value_counts().plot(kind='bar')
        plt.title('Sentiment Distribution')
        
        # Top hashtags
        plt.subplot(2, 2, 2)
        hashtag_counts.plot(kind='barh')
        plt.title('Top 20 Hashtags')
        
        # Word cloud
        plt.subplot(2, 2, 3)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Tweets')
        
        # Save visualizations
        plt.tight_layout()
        plt.savefig('scraped_results/analysis_visualizations.png')
        plt.close()
        
        # Save analysis report
        with open('scraped_results/analysis_report.txt', 'w') as f:
            f.write("Twitter Data Analysis Report\n")
            f.write("="*50 + "\n\n")
            
            f.write("1. Sentiment Analysis\n")
            f.write(str(analysis_results['sentiment_distribution']) + "\n\n")
            
            f.write("2. Top Hashtags\n")
            f.write(str(analysis_results['top_hashtags']) + "\n\n")
            
            f.write("3. Engagement Statistics\n")
            f.write(str(analysis_results['engagement_stats']) + "\n\n")
            
            f.write("Key Insights:\n")
            f.write("- The majority of tweets are " + 
                  analysis_results['sentiment_distribution'].idxmax() + "\n")
            f.write("- Most popular hashtags: " + 
                  ', '.join(analysis_results['top_hashtags'].head(5).index.tolist()) + "\n")
            f.write("- Average engagement: " + 
                  str(analysis_results['engagement_stats'].loc['mean'].round(1).to_dict()) + "\n")
            
        logging.info("Analysis completed and saved to scraped_results/")

if __name__ == "__main__":
    print("""
    Twitter Scraper for Quant Research Analyst Position
    --------------------------------------------------
    This program will:
    1. Scrape at least 2000 tweets with job-related hashtags
    2. Clean and preprocess the data
    3. Perform sentiment analysis and other insights
    4. Save results to CSV and generate analysis report
    
    Instructions:
    1. A Chrome browser will open (not incognito)
    2. If prompted, log in to Twitter manually
    3. The script will automatically collect tweets
    4. Press Enter in the console when done to close
    """)
    
    scraper = TwitterScraper()
    scraper.scrape_tweets()
    scraper.analyze_tweets()
