import json
import random
import os
import re
import requests
from bs4 import BeautifulSoup

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.fixed_file = os.path.join(self.data_dir, "fixed_urls.json")
        self.scraped_fixed_file = os.path.join(self.data_dir, "scraped_fixed.json")
        self.scraped_random_file = os.path.join(self.data_dir, "scraped_random.json")
        self.scraped_all_file = os.path.join(self.data_dir, "scraped_all.json")
        self.random_file = os.path.join(self.data_dir, "random_urls.json")

    def clean_text(self, text):
        text = re.sub(r"\\[[0-9]+\\]", "", text)
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    def chunk_text(self, text, chunk_size=300, overlap=50):
        """
        Token-based chunking with overlap. Default 300 tokens, 50 overlap.
        """
        tokens = text.split()
        if not tokens:
            return []
        chunks = []
        step = max(1, chunk_size - overlap)
        for start in range(0, len(tokens), step):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                continue
            chunks.append(" ".join(chunk_tokens))
            if end == len(tokens):
                break
        return chunks

    def scrape_content(self, url):
        """
        Fetches the URL and extracts the full text from paragraph tags.
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
            
            for p in paragraphs:
                text = p.get_text().strip()
                if text:
                    text_content.append(text)
            
            full_text = " ".join(text_content)
            return self.clean_text(full_text)
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""

    def build_article_record(self, url, title, text, source, url_id):
        chunks = self.chunk_text(text, chunk_size=300, overlap=50)
        chunk_records = []
        for idx, chunk_text in enumerate(chunks, start=1):
            chunk_id = f"{url_id}_{source}_chunk{idx}"
            chunk_records.append({
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "text": chunk_text
            })
        return {
            "source": source,
            "url_id": url_id,
            "url": url,
            "title": title,
            "text": text,
            "chunks": chunk_records
        }

    def _load_cached_articles(self, path):
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        if data and isinstance(data, list) and "chunks" in data[0]:
            return data
        return None

    def load_fixed_data(self):
        """
        Loads fixed data (article + chunks). Checks cache first, else scrapes.
        """
        cached = self._load_cached_articles(self.scraped_fixed_file)
        if cached is not None:
            print(f"Loading cached fixed data from {self.scraped_fixed_file}")
            return cached
        
        print(f"Scraping fixed URLs from {self.fixed_file}...")
        if os.path.exists(self.fixed_file):
            with open(self.fixed_file, 'r') as f:
                input_data = json.load(f)
                urls = input_data.get("urls", [])
                
            scraped_data = []
            for i, url in enumerate(urls, start=1):
                print(f"Scraping {url}...")
                title = url.split("/")[-1].replace("_", " ")
                text = self.scrape_content(url)
                if text:
                    scraped_data.append(self.build_article_record(url, title, text, "fixed", i))
            
            # Cache the result
            with open(self.scraped_fixed_file, 'w') as f:
                json.dump(scraped_data, f, indent=4)
                
            return scraped_data
        else:
            print("Fixed URLs file not found.")
            return []

    def fetch_random_data(self):
        """
        Loads random URLs from file and scrapes them (article + chunks).
        """
        cached = self._load_cached_articles(self.scraped_random_file)
        if cached is not None:
            print(f"Loading cached random data from {self.scraped_random_file}")
            return cached

        print(f"Scraping random URLs from {self.random_file}...")
        if os.path.exists(self.random_file):
            with open(self.random_file, 'r') as f:
                input_data = json.load(f)
                urls = input_data.get("urls", [])
            
            scraped_data = []
            for i, url in enumerate(urls, start=1):
                print(f"Scraping {url}...")
                title = url.split("/")[-1].replace("_", " ")
                text = self.scrape_content(url)
                if text:
                    scraped_data.append(self.build_article_record(url, title, text, "random", i))

            with open(self.scraped_random_file, 'w') as f:
                json.dump(scraped_data, f, indent=4)
            return scraped_data
        else:
            print("Random URLs file not found.")
            return []

    def flatten_chunks(self, articles):
        chunks = []
        for article in articles:
            for chunk in article.get("chunks", []):
                chunks.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "url": article.get("url"),
                    "title": article.get("title"),
                    "text": chunk.get("text"),
                    "source": article.get("source"),
                    "url_id": article.get("url_id")
                })
        return chunks

    def load_all_data(self):
        cached = self._load_cached_articles(self.scraped_all_file)
        if cached is not None:
            print(f"Loading cached combined data from {self.scraped_all_file}")
            return self.flatten_chunks(cached)

        fixed = self.load_fixed_data()
        random_set = self.fetch_random_data()
        combined = fixed + random_set

        with open(self.scraped_all_file, 'w') as f:
            json.dump(combined, f, indent=4)

        return self.flatten_chunks(combined)
