import json
import random
import os
import requests
from bs4 import BeautifulSoup

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.fixed_file = os.path.join(self.data_dir, "fixed_urls.json")
        self.scraped_fixed_file = os.path.join(self.data_dir, "scraped_fixed.json")
        self.random_file = os.path.join(self.data_dir, "random_urls.json")

    def scrape_content(self, url):
        """
        Fetches the URL and extracts the first 200 words from paragraph tags.
        """
        try:
            headers = {
                "User-Agent": "MyRAGApp/1.0 (contact@example.com)"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text from p tags
            paragraphs = soup.find_all('p')
            text_content = []
            word_count = 0
            
            for p in paragraphs:
                text = p.get_text().strip()
                if text:
                    words = text.split()
                    text_content.append(text)
                    word_count += len(words)
                    if word_count >= 200:
                        break
            
            full_text = " ".join(text_content)
            # Truncate to strictly 200 words if slightly over
            all_words = full_text.split()
            return " ".join(all_words[:200])
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""

    def load_fixed_data(self):
        """
        Loads fixed data. Checks cache first, else scrapes.
        """
        if os.path.exists(self.scraped_fixed_file):
            print(f"Loading cached fixed data from {self.scraped_fixed_file}")
            with open(self.scraped_fixed_file, 'r') as f:
                return json.load(f)
        
        print(f"Scraping fixed URLs from {self.fixed_file}...")
        if os.path.exists(self.fixed_file):
            with open(self.fixed_file, 'r') as f:
                input_data = json.load(f)
                urls = input_data.get("urls", [])
                
            scraped_data = []
            for url in urls:
                print(f"Scraping {url}...")
                title = url.split("/")[-1].replace("_", " ")
                text = self.scrape_content(url)
                if text:
                    scraped_data.append({
                        "url": url,
                        "title": title,
                        "text": text
                    })
            
            # Cache the result
            with open(self.scraped_fixed_file, 'w') as f:
                json.dump(scraped_data, f, indent=4)
                
            return scraped_data
        else:
            print("Fixed URLs file not found.")
            return []

    def fetch_random_data(self):
        """
        Loads random URLs from file and scrapes them fresh on every call.
        """
        print(f"Scraping random URLs from {self.random_file}...")
        if os.path.exists(self.random_file):
            with open(self.random_file, 'r') as f:
                input_data = json.load(f)
                urls = input_data.get("urls", [])
            
            scraped_data = []
            for url in urls:
                print(f"Scraping {url}...")
                title = url.split("/")[-1].replace("_", " ")
                text = self.scrape_content(url)
                if text:
                    scraped_data.append({
                        "url": url,
                        "title": title,
                        "text": text
                    })
            return scraped_data
        else:
             print("Random URLs file not found.")
             return []

    def load_all_data(self):
        fixed = self.load_fixed_data()
        random_set = self.fetch_random_data()
        return fixed + random_set
