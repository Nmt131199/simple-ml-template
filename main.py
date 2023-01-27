# import libraries
import pandas as pd

from processing.preprocess import remove_noise, fill_na
from processing.features_engineering import encode_data, scale_data

from utils.split_data import split_train_test
from utils.training import train_model
from utils.predicting import predict

from models.params import setup_params
from models.cls_models import build_model

# load data
data = pd.read_csv('dog_vs_cat.csv')

# preprocess
data = remove_noise(data)
data = fill_na(data)

# feature engineering
data = encode_data(data)
data = scale_data(data)

# split data
X_train, y_train, X_test, y_test = split_train_test(data)

# train model
# set up
params = setup_params()
model = build_model(params)

# train
model = train_model(model=model, train_data=(X_train, y_train))

