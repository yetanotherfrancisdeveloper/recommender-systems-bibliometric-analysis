import requests
import wget
from bs4 import BeautifulSoup
from config import PDF_PATH
from tqdm import tqdm
from utils import read_data


def find_pdf(df):
    for link in tqdm(df.entry_id):
        # Get and read page in BeautifulSoup
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        # Download and save page
        page_pdf_download = soup.find_all('a', class_='abs-button download-pdf')
        try:
            pdf_address = 'https://arxiv.org' + page_pdf_download[0]['href'] + '.pdf'
        except IndexError:
            print(f'There is nothing to download for {link}')
            continue
        try:
            wget.download(pdf_address, PDF_PATH)
        except (ValueError, ConnectionResetError):
            print(f'This url does not exist: {pdf_address}')
            continue


if __name__ == '__main__':
    data = read_data()
    find_pdf(data)
