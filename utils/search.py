import requests
from config.keys import SERPAPI_API_KEY

def web_search(query, num_results=3):
    url = "https://api.example-search.com/search"
    headers = {"Authorization": f"Bearer {SERPAPI_API_KEY}"}
    params = {"q": query, "num": num_results}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return [item['snippet'] for item in response.json().get('results', [])]
    else:
        return ["No results found."]
