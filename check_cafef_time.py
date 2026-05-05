# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import re

url = 'https://cafef.vn/sep-vpbank-muon-mua-30-trieu-co-phieu-vpb-gia-tri-840-ty-dong-188260505145042849.chn'

try:
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
    soup = BeautifulSoup(r.text, 'html.parser')

    # Find span/div with time/date
    time_tags = soup.find_all(['span', 'div'], class_=lambda x: x and ('time' in x or 'date' in x))

    print("Time tags found:")
    for tag in time_tags[:5]:
        text = tag.get_text(strip=True)[:100]
        classes = tag.get('class', [])
        print(f"  {classes}: {text}")

    # Fallback: find dates in text
    all_text = soup.get_text()
    dates = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}', all_text)
    if dates:
        print(f"Dates in text: {set(dates)}")

except Exception as e:
    print(f"Error: {e}")
