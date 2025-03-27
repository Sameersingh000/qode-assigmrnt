from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Setup WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")  # Less detectable headless mode
options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent bot detection
options.add_argument("--log-level=3")  # Suppress logs

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open Twitter/X
driver.get("https://twitter.com/explore")

try:
    # Wait for the search box to appear
    wait = WebDriverWait(driver, 10)
    search_box = wait.until(EC.presence_of_element_located((By.XPATH, "//input[contains(@placeholder, 'Search')]")))
    
    # Enter search query
    search_query = "#naukri OR #jobs OR #jobseeker OR #vacancy since:2024-01-01 until:2025-03-27"
    search_box.send_keys(search_query)
    search_box.send_keys(Keys.RETURN)

    # Wait for results to load
    time.sleep(5)

    # Scroll to load more tweets
    for _ in range(3):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

    # Scrape tweets
    tweets_data = []
    tweets = driver.find_elements(By.XPATH, "//article[@data-testid='tweet']")

    for tweet in tweets:
        try:
            username = tweet.find_element(By.XPATH, ".//div[@dir='ltr']").text
            tweet_text = tweet.find_element(By.XPATH, ".//div[@data-testid='tweetText']").text
            timestamp = tweet.find_element(By.XPATH, ".//time").get_attribute("datetime")

            tweets_data.append([username, tweet_text, timestamp])
        except Exception:
            continue  # Skip errors

    # Save to CSV
    df = pd.DataFrame(tweets_data, columns=["Username", "Tweet", "Timestamp"])
    df.to_csv("tweets.csv", index=False, encoding="utf-8")

    print("✅ Tweets saved to tweets.csv!")

except Exception as e:
    print(f"❌ Error: {e}")

finally:
    driver.quit()  # Close browser
