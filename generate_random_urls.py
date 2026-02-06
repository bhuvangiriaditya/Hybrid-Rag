import requests
import json
import os

def get_random_wikipedia_urls(n=300):
    base_api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": n,
        "format": "json"
    }
    
    headers = {
        "User-Agent": "MyRAGApp/1.0 (contact@example.com)"
    }

    try:
        response = requests.get(base_api, params=params, headers=headers).json()
        random_pages = response["query"]["random"]

        urls = [
            f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}"
            for item in random_pages
        ]

        return urls
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

if __name__ == "__main__":
    print("Fetching random Wikipedia URLs...")
    random_urls = get_random_wikipedia_urls(300)
    
    output_dir = "data"
    output_file = os.path.join(output_dir, "random_urls.json")
    
    # Ensure directory exists (though we checked it does)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({"total_urls": len(random_urls), "urls": random_urls}, f, indent=4)
    
    print(f"Successfully saved {len(random_urls)} URLs to {output_file}")
