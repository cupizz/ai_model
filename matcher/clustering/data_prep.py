import os
from pathlib import Path

import regex as re
from pyvi import ViTokenizer

from matcher.clustering import utils

EMAIL = re.compile(r"([\w0-9_\.-]+)(@)([\d\w\.-]+)(\.)([\w\.]{2,6})")
URL = re.compile(r"https?:\/\/(?!.*:\/\/)\S+")
PHONE = re.compile(r"(09|01[2|6|8|9])+([0-9]{8})\b")
MENTION = re.compile(r"@.+?:")
NUMBER = re.compile(r"\d+.?\d*")
DATETIME = '\d{1,2}\s?[/-]\s?\d{1,2}\s?[/-]\s?\d{4}'

RE_HTML_TAG = re.compile(r'<[^>]+>')
RE_CLEAR_1 = re.compile("[^_<>\s\p{Latin}]")
RE_CLEAR_2 = re.compile("__+")
RE_CLEAR_3 = re.compile("\s+")

stopwords: str = []

base_dir = Path(__file__).parent

vn_path = os.path.join(base_dir, "vietnamese-stopwords.txt")


def load_stopwords_vietnamese():
    with open(vn_path, 'r') as file:
        # reading each line     
        for line in file:
            # reading each word         
            for word in line.split():
                if word not in stopwords:
                    stopwords.append(word)


class TextPreprocess:
    @staticmethod
    def replace_common_token(txt):
        txt = re.sub(EMAIL, ' ', txt)
        txt = re.sub(URL, ' ', txt)
        txt = re.sub(MENTION, ' ', txt)
        txt = re.sub(DATETIME, ' ', txt)
        txt = re.sub(NUMBER, ' ', txt)
        return txt

    @staticmethod
    def remove_emoji(txt):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', txt)

    @staticmethod
    def remove_html_tag(txt):
        return re.sub(RE_HTML_TAG, ' ', txt)

    @staticmethod
    def remove_vietnamese_stopwords(txt):
        if len(stopwords) == 0:
            load_stopwords_vietnamese()
        # Splitting on spaces between words
        txt = txt.split(' ')
        txt = [i for i in txt if i not in stopwords]
        return ' '.join(txt)

    @staticmethod
    def preprocess(txt, tokenize=True):
        txt = txt.lower()
        txt = TextPreprocess.remove_html_tag(txt)
        txt = re.sub('&.{3,4};', ' ', txt)
        txt = utils.convertwindown1525toutf8(txt)
        if tokenize:
            txt = ViTokenizer.tokenize(txt)
        txt = TextPreprocess.replace_common_token(txt)
        txt = TextPreprocess.remove_emoji(txt)
        txt = re.sub(RE_CLEAR_1, ' ', txt)
        txt = re.sub(RE_CLEAR_2, ' ', txt)
        txt = re.sub(RE_CLEAR_3, ' ', txt)
        txt = TextPreprocess.remove_vietnamese_stopwords(txt)
        txt = utils.chuan_hoa_dau_cau_tieng_viet(txt)
        return txt.strip()
