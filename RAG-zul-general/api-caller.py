import requests

API_KEY = "d2kfc19r01qs23a21bbgd2kfc19r01qs23a21bc0"
url = f"https://finnhub.io/api/v1/news?category=general&token={API_KEY}"

response = requests.get(url)

if response.status_code == 200:
    news_data = response.json()
    print(len(news_data))
    for article in news_data[:5]:  # Show top 5 latest articles
        print(f"Headline: {article['headline']}")
        print(f"Source: {article['source']}")
        print(f"URL: {article['url']}\n")
else:
    print("Error:", response.status_code, response.text)
