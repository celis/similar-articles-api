import boto3
from gensim.models.keyedvectors import KeyedVectors


class PythonPredictor:

    MODEL = "article_embeddings.bin"

    def __init__(self, config):
        s3 = boto3.client("s3")
        s3.download_file(config["bucket"], config["key"], self.MODEL)
        self.model = KeyedVectors.load_word2vec_format(self.MODEL, binary=True)

    def predict(self, payload):
        id = payload["id"]
        predictions = self._most_similar(id)
        return predictions

    def _most_similar(self, article: str, topn: int = 5):
        """
        Predicts most similar items
        """
        return [article[0] for article in self.model.similar_by_word(article, topn)]