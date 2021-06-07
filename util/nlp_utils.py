from pandas.core import frame
import matplotlib.pyplot as plt
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import stopwords
from operator import itemgetter
import pandas as pd
from wordcloud import WordCloud
from pathlib import Path
import imageio
import spacy


class NlpUtils:

    @staticmethod
    def top_n_words(top_n: int, text: str, language='english') -> frame:
        """
        Top N words in the text
        :param top_n: number of words to return
        :param text: text content
        :param language: text's language, default='english'
        :return: top N words in text
        """
        stop_words = stopwords.words(language)
        text_blob = TextBlob(text)
        text_blob_items = text_blob.word_counts.items()
        text_blob_items = [item for item in text_blob_items if item[0] not in stop_words]
        text_blob_items_sorted = sorted(text_blob_items, key=itemgetter(1), reverse=True)
        top_n_items = text_blob_items_sorted[1:top_n + 1]
        data_frame = pd.DataFrame(top_n_items, columns=['Word', 'Count'])
        return data_frame

    @staticmethod
    def top_n_words_example(top_n: int):
        text = Path('resources/RomeoAndJuliet.txt').read_text()
        df = NlpUtils.top_n_words(top_n, text)
        plt.axes = df.plot.bar(x='Word', y='Count', legend=False)
        plt.gcf().tight_layout()
        plt.show()

    @staticmethod
    def word_cloud_generator(text: str, mask_image: Path, save_to_file=False) -> WordCloud:
        """
        Generates word cloud of a text as an image
        :param text: text content
        :param mask_image: shape of the word cloud: address of a png image file
        :param save_to_file: whether save final image to disk or not
        :return: word cloud image
        """
        mask = imageio.imread(mask_image)
        word_cloud = WordCloud(colormap='prism', mask=mask, background_color='white')
        word_cloud = word_cloud.generate(text)
        if save_to_file:
            word_cloud.to_file('word_cloud.png')
        return word_cloud

    @staticmethod
    def word_cloud_generator_example():
        word_cloud_text = Path('resources/RomeoAndJuliet.txt').read_text()
        mask_image = Path('resources/mask_heart.png')
        word_cloud = NlpUtils.word_cloud_generator(word_cloud_text, mask_image, True)
        plt.imshow(word_cloud)
        plt.axis('off')
        plt.show()

    @staticmethod
    def naive_sentiment_analyzer(text: str) -> dict:
        """
        Sentiment analyzer for every sentence in a text and overall sentiment
        :param text: text content to sentiment analyze
        :return: dictionary of sentences and sentiments
        """
        result = {}
        text_blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        result['overall'] = text_blob.sentiment
        for sentence in text_blob.sentences:
            result[sentence] = sentence.sentiment

        return result

    @staticmethod
    def naive_sentiment_analyzer_example():
        text = "My sister & I were excited to go see it after watching the trailer. " \
               "The trailer was better than the movie! There were some believable scenes " \
               "but so many times we would look at each other and say, “You’re kidding!” " \
               "When you saw Indians they never made a sound & seemingly deaf, trudging along " \
               "as if in a daze.  I doubt townsfolk would be giving the Hanks character a " \
               "standing ovation that went on & on & on with every single person grinning " \
               "from ear to ear. I could’ve left after 30 minutes but for some reason I sat " \
               "through it. It was all predictable & Tom Hanks or not, so painfully boring."
        result = NlpUtils.naive_sentiment_analyzer(text)
        for item in result:
            print(
                f'{item} \n{result[item].classification} | p_pos={result[item].p_pos:.3f}, p_neg={result[item].p_neg:.3f}\n')

    @staticmethod
    def entity_recognition(text: str) -> spacy:
        """
        Entity recognition from text file
        :param text: text content for entity recognition
        :return: entities
        """
        nlp = spacy.load('en_core_web_sm')
        document = nlp(text)
        return document

    @staticmethod
    def entity_recognition_example():
        text = "From 1960 to 1996, Guatemala endured a bloody civil war fought " \
               "between the US-backed government and leftist rebels, including " \
               "genocidal massacres of the Maya population perpetrated by the military." \
               "Since a United Nations-negotiated peace accord, Guatemala has achieved " \
               "both economic growth and successful democratic elections, although it " \
               "continues to struggle with high rates of poverty and crime, drug cartels, " \
               "and instability. As of 2014, Guatemala ranks 31st of 33 Latin American and " \
               "Caribbean countries in terms of the Human Development Index."
        document = NlpUtils.entity_recognition(text)
        for entity in document.ents:
            print(f'{entity.text}: {entity.label_}')

    @staticmethod
    def similarity_detection(text_one: str, text_two: str) -> float:
        """

        :param text_one: text one to compare with text two
        :param text_two: second text content
        :return: similarity percentage
        """
        nlp = spacy.load('en_core_web_sm')
        document_one = nlp(text_one)
        document_two = nlp(text_two)
        similarity = document_one.similarity(document_two)
        return similarity

    @staticmethod
    def similarity_detection_example():
        text_one = Path('resources/RomeoAndJuliet.txt').read_text()
        text_two = Path('resources/EdwardTheSecond.txt').read_text()
        similarity = NlpUtils.similarity_detection(text_one, text_two)
        print(f'similarity: {similarity}')
