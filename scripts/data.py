import nltk
import re
import numpy as np
import pandas as pd
import ray

from typing import Dict, List, Tuple
from ray.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from nltk.corpus import stopwords
 
# nltk.download('stopwords') ?? maybe not needed?
STOPWORDS = set(stopwords.words('english'))

def load_data(dataset_loc: str, num_samples: int = None) -> Dataset:
    """Load data from source into a Ray Dataset.

    Args:
        dataset_loc (str): Location of the dataset.
        num_samples (int, optional): The number of samples to load. Defaults to None.

    Returns:
        Dataset: Our dataset represented by a Ray Dataset.
    """
    ds = ray.data.read_csv(dataset_loc)
    ds = ds.random_shuffle(seed=42)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds
    return ds

def clean_text(text: str, stopwords: List = STOPWORDS) -> str:
    """Clean raw text string.

    Args:
        text (str): Raw text to clean.
        stopwords (List, optional): list of words to filter out. Defaults to STOPWORDS.

    Returns:
        str: cleaned text.
    """
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  # remove links

    return text

def tokenize(batch: Dict) -> Dict:
    """Tokenize the text input in our batch using a tokenizer.

    Args:
        batch (Dict): batch of data with the text inputs to tokenize.

    Returns:
        Dict: batch of data with the results of tokenization (`input_ids` and `attention_mask`) on the text inputs.
    """
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["tag"]))

def preprocess(df: pd.DataFrame, class_to_index: Dict) -> Dict:
    """Preprocess the data in our dataframe.

    Args:
        df (pd.DataFrame): Raw dataframe to preprocess.
        class_to_index (Dict): Mapping of class names to indices.

    Returns:
        Dict: preprocessed data (ids, masks, targets).
    """
    df["Article"] = df.Article.apply(clean_text)  # clean text
    df = df.drop(columns=["ArticleId"], errors="ignore")  # clean dataframe
    df = df[["Article", "Category"]]  
    df["Category"] = df["Category"].map(class_to_index)  # label encoding
    outputs = tokenize(df)
    return outputs

class CustomPreprocessor:
    """Custom preprocessor class."""

    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    def fit(self, ds):
        categories = ds.unique(column="Cateogry")
        self.class_to_index = {tag: i for i, tag in enumerate(categories)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        return self

    def transform(self, ds):
        return ds.map_batches(preprocess, fn_kwargs={"class_to_index": self.class_to_index}, batch_format="pandas")