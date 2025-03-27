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
from pathlib import Path
import random

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_scraper.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

class TwitterScraper:
    def __init__(self):
        self.tweets_data = []
        self.hashtags = ["naukri", "jobs", "jobseeker", "vacancy"]
        self.target_count = 2000
        self.scraped_count = 0
        self.timeout = 300000  # 5 minutes in milliseconds
        self.max_scrolls = 150
        self.user_data_dir = str(Path.home() / "twitter_scraper_data")
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = word_tokenize(text)
        words = [word for word in words if word not in stopwords.words('english')]
        return ' '.join(words)

    def scrape_tweets(self):
        with sync_playwright() as pw:
            try:
                # Launch browser with enhanced configuration
                browser = pw.chromium.launch_persistent_context(
                    self.user_data_dir,
                    headless=False,
                    channel="chrome",
                    viewport={"width": 1280, "height": 800},
                    user_agent=random.choice(self.user_agents),
                    timeout=self.timeout,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--start-maximized"
                    ],
                    ignore_https_errors=True
                )
                
                page = browser.new_page()
                page.set_default_timeout(self.timeout)

                # Randomize browsing patterns
                page.set_extra_http_headers({
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/'
                })

                for hashtag in self.hashtags:
                    if self.scraped_count >= self.target_count:
                        break

                    logging.info(f"Processing #{hashtag}...")
                    try:
                        # Random delay before navigation
                        time.sleep(random.uniform(1, 3))
                        
                        page.goto(
                            f"https://twitter.com/search?q=%23{hashtag}&src=typed_query&f=live",
                            wait_until="networkidle",
                            timeout=60000
                        )
                        
                        # Check for login wall
                        if page.is_visible("text='Log in'", timeout=5000):
                            logging.info("Login required - please authenticate in the browser window")
                            page.wait_for_selector("a[href='/home']", timeout=120000)
                        
                        # Wait for tweets with multiple fallback selectors
                        try:
                            page.wait_for_selector('article[data-testid="tweet"]', timeout=60000)
                        except:
                            page.wait_for_selector('div[data-testid="tweet"]', timeout=30000)
                        
                        logging.info(f"Found initial tweets for #{hashtag}")

                        scroll_count = 0
                        last_count = 0
                        stale_count = 0

                        while (self.scraped_count < self.target_count and 
                               scroll_count < self.max_scrolls):
                            
                            # Get current tweets
                            tweets = page.query_selector_all('article[data-testid="tweet"]') or \
                                     page.query_selector_all('div[data-testid="tweet"]')
                            current_count = len(tweets)
                            
                            if current_count > last_count:
                                new_tweets = tweets[last_count:current_count]
                                self.process_new_tweets(new_tweets)
                                last_count = current_count
                                stale_count = 0
                            else:
                                stale_count += 1
                                if stale_count > 5:
                                    logging.warning("No new tweets detected - breaking scroll loop")
                                    break

                            # Human-like scrolling
                            scroll_distance = random.randint(500, 1000)
                            page.evaluate(f"window.scrollBy(0, {scroll_distance})")
                            scroll_count += 1
                            
                            # Randomized delay
                            time.sleep(random.uniform(2, 5))
                            logging.info(f"Scroll {scroll_count}: Collected {self.scraped_count}/{self.target_count} tweets")

                    except Exception as e:
                        logging.error(f"Error processing #{hashtag}: {str(e)}")
                        page.screenshot(path=f"error_{hashtag}_{int(time.time())}.png")
                        continue

            except Exception as e:
                logging.error(f"Fatal error: {str(e)}")
                if 'page' in locals():
                    page.screenshot(path=f"fatal_error_{int(time.time())}.png")
            finally:
                if 'browser' in locals():
                    input("Press Enter to close the browser...")
                    browser.close()

    def process_new_tweets(self, tweets):
        for tweet in tweets:
            try:
                # Extract username with fallback
                username_element = tweet.query_selector("div[data-testid='User-Name']") or \
                                  tweet.query_selector("div[data-testid='userName']")
                username = username_element.inner_text().strip() if username_element else "Unknown"
                
                # Extract content with fallback
                content_element = tweet.query_selector("div[data-testid='tweetText']") or \
                                tweet.query_selector("div[lang]")
                content = content_element.inner_text().strip() if content_element else ""
                
                # Extract datetime with fallback
                time_element = tweet.query_selector("time")
                datetime_str = time_element.get_attribute("datetime") if time_element else ""
                date, time = datetime_str.split('T')[0], datetime_str.split('T')[1].split('.')[0] if datetime_str else ("", "")
                
                # Extract metrics with improved reliability
                metrics = {
                    'likes': self.get_metric(tweet, "[data-testid='like']"),
                    'retweets': self.get_metric(tweet, "[data-testid='retweet']"),
                    'comments': self.get_metric(tweet, "[data-testid='reply']"),
                    'views': self.get_metric(tweet, "[aria-label*='view']", is_aria=True),
                    'replies': self.get_metric(tweet, "[aria-label*='reply']", is_aria=True)
                }

                self.tweets_data.append([
                    username,
                    content,
                    date,
                    time,
                    ', '.join(re.findall(r'@\w+', content)),
                    ', '.join(re.findall(r'#\w+', content)),
                    metrics['likes'],
                    metrics['retweets'],
                    metrics['comments'],
                    metrics['replies'],
                    metrics['views']
                ])
                self.scraped_count += 1

            except Exception as e:
                logging.warning(f"Skipping tweet: {str(e)}")
                continue

    def get_metric(self, element, selector, is_aria=False):
        try:
            target = element.query_selector(selector)
            if not target:
                return "0"
            
            if is_aria:
                aria_label = target.get_attribute("aria-label")
                return re.search(r'\d+', aria_label).group() if aria_label else "0"
            return target.inner_text()
        except:
            return "0"

    def save_to_csv(self):
        try:
            df = pd.DataFrame(self.tweets_data, columns=[
                "Username", "Tweet", "Date", "Time", 
                "Mentions", "Hashtags", "Likes", 
                "Retweets", "Comments", "Replies", "Views"
            ])
            
            # Convert numeric columns
            numeric_cols = ["Likes", "Retweets", "Comments", "Replies", "Views"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tweets_data_{timestamp}.csv"
            df.to_csv(filename, index=False)
            
            logging.info(f"Successfully saved {len(df)} tweets to {filename}")
            return df
        except Exception as e:
            logging.error(f"Failed to save CSV: {str(e)}")
            return None

if __name__ == "__main__":
    print("""
    Enhanced Twitter Scraper
    ========================
    1. Will open Chrome browser (your regular profile)
    2. If login required, please authenticate manually
    3. Scraping will begin automatically
    4. Press Enter in console when done to close browser
    """)
    
    scraper = TwitterScraper()
    try:
        scraper.scrape_tweets()
        scraper.save_to_csv()
    except KeyboardInterrupt:
        logging.info("Script interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        logging.info("Scraping session ended")
