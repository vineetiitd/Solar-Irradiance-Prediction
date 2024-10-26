# Importing Libraries
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

import xgboost as xgb

from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential
from collections import Counter

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Loading Data
data = pd.read_csv("solar_irradiance data.csv")


# Data Wrangling
df = data.copy()
# extract the date from the date_time format of the 'Data' parameter
df["Data"] = df["Data"].apply(lambda x: x.split()[0])

df["Month"] = pd.to_datetime(df["Data"]).dt.month
df["Day"] = pd.to_datetime(df["Data"]).dt.day
df["Hour"] = pd.to_datetime(df["Time"]).dt.hour
df["Minute"] = pd.to_datetime(df["Time"]).dt.minute
df["Second"] = pd.to_datetime(df["Time"]).dt.second

df["risehour"] = (
    df["TimeSunRise"].apply(lambda x: re.search(r"^\d+", x).group(0)).astype(int)
)
df["riseminuter"] = (
    df["TimeSunRise"]
    .apply(lambda x: re.search(r"(?<=\:)\d+(?=\:)", x).group(0))
    .astype(int)
)

df["sethour"] = (
    df["TimeSunSet"].apply(lambda x: re.search(r"^\d+", x).group(0)).astype(int)
)
df["setminute"] = (
    df["TimeSunSet"]
    .apply(lambda x: re.search(r"(?<=\:)\d+(?=\:)", x).group(0))
    .astype(int)
)

df.drop(["UNIXTime", "Data", "Time", "TimeSunRise", "TimeSunSet"], axis=1, inplace=True)

# checking for null values in the data
df.isnull().sum().sum()


# Feature Selection using Correlation Matrix
corr_matrix = df.corr()
# plot the correlation matrix using heatmap for clear understanding
plt.figure(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True)
plt.show()


# Feature Selection using SelectKBest Method
# use the label encoder
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
train_Y = label_encoder.fit_transform(target)
target_cont = df["Radiation"].apply(lambda x: int(x * 100))
scaled_input_features = MinMaxScaler().fit_transform(input_features)
fit = bestfeatures.fit(scaled_input_features, target_cont)
scores = pd.DataFrame(fit.scores_)
column = pd.DataFrame(input_features.columns)
# contatinating data_features with the scores
featureScores = pd.concat([column, scores], axis=1)
featureScores.columns = ["Features", "feature_imp"]
featureScores.sort_values(by="feature_imp", ascending=False, inplace=True)


# Feature Selection using Extra Tree Classifier
model = ExtraTreesClassifier(verbose=2, n_estimators=10)
model.fit(scaled_input_features, target_cont)
feature_importances = pd.DataFrame(
    model.feature_importances_, index=input_features.columns, columns=["feature_imp"]
)
feature_importances.sort_values(by="feature_imp", ascending=False, inplace=True)


# Feature Engineering with BoxCox, Log, Min-Max and Standard transformation
features_to_transform = [
    "Temperature",
    "Pressure",
    "Humidity",
    "Speed",
    "WindDirection(Degrees)",
]

for i in features_to_transform:

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 5))

    pd.DataFrame(input_features[i]).hist(ax=ax1, bins=50)
    pd.DataFrame((input_features[i] + 1).transform(np.log)).hist(ax=ax2, bins=50)
    pd.DataFrame(stats.boxcox(input_features[i] + 1)[0]).hist(ax=ax3, bins=50)
    pd.DataFrame(
        StandardScaler().fit_transform(np.array(input_features[i]).reshape(-1, 1))
    ).hist(ax=ax4, bins=50)
    pd.DataFrame(
        MinMaxScaler().fit_transform(np.array(input_features[i]).reshape(-1, 1))
    ).hist(ax=ax5, bins=50)

    ax1.set_ylabel("Normal")
    ax2.set_ylabel("Log")
    ax3.set_ylabel("Box Cox")
    ax4.set_ylabel("Standard")
    ax5.set_ylabel("MinMax")

# set the transformations required
transform = {
    "Temperature": (input_features["Temperature"] + 1).transform(np.log),
    "Pressure": stats.boxcox(input_features["Pressure"] + 1)[0],
    "Humidity": stats.boxcox(input_features["Humidity"] + 1)[0],
    "Speed": (input_features["Speed"] + 1).transform(np.log),
    "WindDirection(Degrees)": MinMaxScaler().fit_transform(
        np.array(input_features["WindDirection(Degrees)"]).reshape(-1, 1)
    ),
}

for i in transform:
    input_features[i] = transform[i]


# Preparing data - Standardisation and Splitting
xtrain, xtest, ytrain, ytest = train_test_split(
    input_features, target, test_size=0.2, random_state=1
)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)


# Prediction with XGBoost
params = {"learning_rate": 0.1, "max_depth": 8}

from xgboost import XGBRegressor

model = XGBRegressor(**params)
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)

rmse = np.sqrt(mean_squared_error(ytest, y_pred))
r2 = r2_score(ytest, y_pred)

print("Testing performance")

print("RMSE: {:.2f}".format(rmse))
print("R2: {:.2f}".format(r2))


# Using MultiLayer Perceptron for prediction
xtrain, xtest, ytrain, ytest = train_test_split(
    input_features, target, test_size=0.2, random_state=1
)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)
model = None
model = Sequential()

model.add(Dense(128, activation="relu", input_dim=14))
model.add(Dropout(0.33))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.33))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.33))

model.add(Dense(1, activation="linear"))

model.compile(metrics="mse", loss="mae", optimizer=Adam(learning_rate=0.001))
print(model.summary())

scores = model.evaluate(xtest, ytest)
mae = scores[0]
mse = scores[1]
print("Mean absolute error: ", mae)
