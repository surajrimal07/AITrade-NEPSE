import requests
from bs4 import BeautifulSoup

def extract_prices():
    url = 'https://www.sharesansar.com/live-trading'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        prices_data = {}
        mu_list = soup.find_all('div', class_='mu-list')
        for mu_item in mu_list:
            name = mu_item.find('h4').text.strip()
            price = mu_item.find('p', class_='mu-price').text.strip()
            index_value = float(price.replace(',', '')) if price else None
            change = mu_item.find('span', class_='mu-percent').text.strip()
            change_value = parse_change(change)
            turnover = remove_commas(price)
            item_data = {
                'Name': name,
                'Index': index_value,
                'Turnover': turnover,
                'Change': change_value
            }
            prices_data[name] = item_data
        return prices_data
    else:
        return {'error': 'Failed to fetch data'}
