import re

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords


# Tokenizing Function
def tokenize_english(text):
    # Instantiating the lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()
    """
    Tokenizing the text, then lemmatizing them
    """
    nltk.download('stopwords')
    nltk.download('wordnet')
    # Creating a library of stopwords
    stops = stopwords.words('english')

    # Lowercasing the words
    text = text.lower()

    regex = r"[a-zA-Z0-9]+"
    matches = re.finditer(regex, text, re.MULTILINE)

    # Splitting on spaces between words
    text = [match.group() for matchNum, match in enumerate(matches, start=1)]

    # Lemmatizing the words and removing stop words
    text = [lemmatizer.lemmatize(i) for i in text if i not in stops]

    return ' '.join(text)


def word_freq(data_frame, column_name):
    total_vocab = set()

    for column in data_frame[column_name]:
        list_words = column.split(' ')
        total_vocab.update(list_words)

    print("Number of unique words: ", len(total_vocab))

    # Determining the most frequent words in user bios
    words = []

    # Adding all the words in each bio to a list
    for column in data_frame[column_name]:
        list_words = column.split(' ')
        words.extend(list_words)

    # Determining the use frequency of each word in all the bios
    bio_freq = nltk.FreqDist(words)
    bio_freq.most_common(104)

    # Plotting the most frequently used words
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 5))

    plt.bar(*zip(*bio_freq.most_common(25)))
    plt.xticks(rotation=75)
    plt.title(f'Most Frequently Used Words in User {column_name}')
    plt.show()

    return total_vocab, words, bio_freq


def bigrams(data_frame):
    total_vocab, words, bio_freq = word_freq(data_frame)
    # Instantiating the score of each bigram
    bigram_meas = nltk.BigramAssocMeasures()

    # Finding and ranking the Bigrams in each bio
    bio_finder = nltk.BigramCollocationFinder.from_words(words)

    # Finding the frequency scores of each bigram
    bio_scored = bio_finder.score_ngrams(bigram_meas.raw_freq)

    # Top 50 most common bigrams
    bio_scored[:50]

    # Creating a list of the bigrams
    bg = list(map(lambda x: x[0][0] + ' ' + x[0][1], bio_scored[:50]))

    # Creating a list of the frequency scores
    bio_scores = list(map(lambda x: x[1], bio_scored[:50]))

    # Combining both the scores and the bigrams
    bigrams = list(zip(bg, bio_scores))

    # Plotting the bigrams and their frequency scores
    plt.style.use('bmh')
    plt.figure(figsize=(15, 5))

    plt.bar(*zip(*bigrams[:25]))
    plt.xticks(rotation=80)
    plt.title('Top 25 Most Common Bigrams')
    plt.show()

    # Filtering out bigrams based on frequency of occurence
    bio_finder.apply_freq_filter(20)

    # Calculating the pointwise mutual information score,
    # which determines how often these words are associated with each other
    bio_pmi = bio_finder.score_ngrams(bigram_meas.pmi)

    print(bio_pmi)

    # Creating bigrams for each pair of words in the bios
    data_frame['Bigrams'] = data_frame.Bios.apply(
        lambda bio: nltk.BigramCollocationFinder.from_words(bio).nbest(bigram_meas.pmi, 100))

    return data_frame
