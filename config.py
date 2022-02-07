import os
import pathlib


MASTER = pathlib.Path().absolute()
DATA_PATH = os.path.join(MASTER, 'data')
PLOT_PATH = os.path.join(MASTER, 'plots')
PDF_PATH = os.path.join(DATA_PATH, 'articles_pdf')
TXT_PATH = os.path.join(DATA_PATH, 'texts')
POS_PATH = os.path.join(DATA_PATH, 'pos_tags.json')
