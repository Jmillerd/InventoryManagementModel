from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic
import pandas as pd


movie_data = Dataset.load_builtin('jester')
