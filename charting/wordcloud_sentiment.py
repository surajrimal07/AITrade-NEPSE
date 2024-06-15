import requests
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt



def fetch_data_from_source(source_url):
    response = requests.get(source_url)
    if response.status_code == 200:
        return response.json()
    else:
        return []

def preprocess_text(text):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('।', ' ')
    text = text.replace('&lsquo;', "'")
    text = text.replace('&rsquo;', "'")
    text = text.replace('&rdquo;', "'")
    text = text.replace('&ndash;', "'")
    return text

def show_sentiment_wordcloud():
    source_urls = [
        f'https://api.zorsha.com.np/news?source=Mero%20Lagani',
        'https://api.zorsha.com.np/news?source=ShareSansar',
        'https://api.zorsha.com.np/news?keyword=nepse'
    ]

    combined_text = ''
    stopwords = set(STOPWORDS)
    stopwords.update(['said','limited','announced', 'becoming','held','much','following','rdquo','ndash'])

    for url in source_urls:
        data = fetch_data_from_source(url)
        combined_text += ' '.join([f"{article['title']} {article['description']}" for article in data])

    combined_text = preprocess_text(combined_text)

    wordcloud = WordCloud(width=800, height=500, stopwords=stopwords, max_words=400, colormap='hot', background_color="white", min_word_length=4).generate(combined_text)

    plt.switch_backend('TkAgg')
    plt.figure(num='Nepse Wordcloud',figsize=(8, 5), facecolor='None')
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

#show_sentiment_wordcloud()
