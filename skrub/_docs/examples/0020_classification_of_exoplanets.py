"""
Classification of Exoplanets
===============================================

In this example, we use a catalog of discovered exoplanets (available from
https://exoplanet.eu/catalog/) to show two classification examples using skrub.
The first is an unsupervised classification using standard skrub functions.
The second is a supervised classification using the DataOps framework.

Since the first detection of 51 Pegasi b* in 1995, thousands of new exoplanets
have been discovered by many observatories, such as Kepler/K2, TESS (Transiting
Exoplanet Satellite Survey), and others. These planets have been detected using
a variety of observational methods, and the planets themselves show a lot of
variety. Let's explore them.

* First detection with a main-sequence host star, (see @article{mayor1995jupiter,
  title={A Jupiter-mass companion to a solar-type star},
  author={Mayor, Michel and Queloz, Didier},
  journal={nature},
  volume={378},
  number={6555},
  pages={355--359},
  year={1995},
  publisher={Nature Publishing Group}
})
"""

# %%
# Initialisation
# ----------------------------------------
# Some tedious but necessary initialisation stuff

import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import skrub

plt.rcParams['font.family'] = 'serif'

# %%
# Loading in the data
# ----------------------------------------
# First load the data into a pandas Dataframe. Some exoplanets have been discovered
# via multiple methods, but for our analysis here we will exclude them. We will also
# exclude any rows where the detection type is labelled as "Other".

fn = "exoplanet.eu_catalog_17-03-26_11_42_20.csv"

df = pd.read_csv(fn)
df = df[~df['detection_type'].str.contains(',', regex=False)]
df = df[df['detection_type'] != 'Other']

# %%
# Let's first take a look at our cleaned dataframe using the TableReport.

skrub.TableReport(df)

# %%
# Data Preprocessing
# ----------------------------------------
# We can use sklearn functions in conjunction with skrub selectors and encoders
# to prepare our data.

from skrub import SelectCols
from skrub import SquashingScaler
from skrub import Cleaner
from skrub import ApplyToCols

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

import skrub.selectors as s

# %%
# We use the SelectCols function to select only the columns we would like to keep
# based on the column name. We first define the transformation, which we name
# `select`, before applying it to our dataframe in a manner similar to a sklearn
# transformation.

target = "detection_type"
columns_to_keep = [
    'mass', 'radius', 'orbital_period', 'eccentricity', 'angular_distance', 'star_distance',
    'star_mass', 'star_radius',
]

features = skrub.var("data", df)

# %%
# First we apply the skrub Cleaner, which performs some basic tidying, before
# scaling our numerical columns in order to homogenize and reduce the dynamic
# range of the values. We also apply the OneHotEncoding encoder to transform our
# string data in numerical data that can be evaluated by a classification algorithm.
# All these transformations can be succinctly packaged using the sklearn
# make_pipeline function, which can accept skrub functions as inputs.

cleaner = Cleaner(
    drop_if_constant=True,
    numeric_dtype="float32",
    drop_if_unique=True,
)

X = (
    features.skb.drop(target)
    .skb.mark_as_X()
    .skb.select(columns_to_keep)
    .skb.apply(cleaner)
    .skb.apply(SquashingScaler(), cols=s.numeric())
    .skb.apply(OneHotEncoder(sparse_output=False), cols=s.string())
)

y = features[target].skb.mark_as_y()


# %%
# Supervised Learning
# ----------------------------------------
# There are many different ways of detecting exoplanets. Which method is most
# effective is dependent on the specifics of the star system. Let's train a
# classifier to predict which detection method would be used to detect a given
# planet based on its mass and orbital period. We will do this using the DataOps
# framework.
#
# To start we define our features (`X`) and target (`y`) via the `skb` functions.
# The functions will return a DataOps object.


# %%
# For the classification we use the HistGradientBoostingClassifier. Our classifier
# variable is constructed from our feature set, and takes our target as an input,
# and thus intrinsically carries information about our dataset.

_hgc = HistGradientBoostingClassifier()
classifier = X.skb.apply(_hgc, y=y)

# %%
# We then define a DataOps learner which will also us to train our classifier in
# a robust way before performing our classification.
#
# In order to verify the effectiveness of our classification, we also split our
# data into training and test sets before fitting. This can be done robustly
# within the DataOps ecosystem, which tracks each split automatically.

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)

learner = classifier.skb.make_learner()

environment = {"data": df}
results = skrub.cross_validate(learner, environment=environment, cv=StratifiedKFold())
print(results)

split = classifier.skb.train_test_split(random_state=1)
split.keys()

learner.fit(split['train'])
predictions = learner.predict(split['test'])

# %%
# Visualize classification results
# ----------------------------------------

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
fig.subplots_adjust(wspace=0.0)

target_names = df['detection_type'].unique()
color_map = dict(zip(matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][:len(target_names)], target_names))
name_to_color = {v: k for k, v in color_map.items()}

colors_test = split['y_test'].map(name_to_color)
ax[0].scatter(split['X_test']['orbital_period'], split['X_test']['mass'], c=colors_test, s=10, alpha=0.65)

colors_train = pd.Series(predictions).map(name_to_color)
ax[1].scatter(split['X_test']['orbital_period'], split['X_test']['mass'], c=colors_train, s=10, alpha=0.65)

ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[0].set_xlabel('mass')
ax[1].set_xlabel('mass')
ax[0].set_title('truth')
ax[1].set_title('prediction')
ax[0].set_ylabel('orbital period')
handles = [ax[1].scatter([], [], c=color, label=name) for name, color in name_to_color.items()]
fig.legend(handles=handles, bbox_to_anchor=(1.01, 0.5), loc='center left')

fig.tight_layout()
plt.show()

# %%
# Let's break down how well our classifier is able to predict the detection type:

for dt in split['y_test'].unique():
    mask = (split['y_test'] == dt)
    truths = split['y_test'][mask]
    predict = predictions[mask]

    print(f"{dt}: {np.sum(predict == truths) / len(truths) * 100:.2f}% accuracy")

# %%
# Unsupervised Learning
# ----------------------------------------
# Now that we have prepared the data let's try performing another type of
# classification, this one unsupervised. Exoplanets come in different sizes and
# locations within their star system (characterised by their mass and orbital
# period), but they tend to fall into broad categories. These categories are not
# strictly defined but we can try to identify some broad trends.

# %%
# The important features pertain to the physical aspects of the exoplanets, thus
# we extract only these features.

X_reduced = df[['orbital_period', 'radius', 'mass', 'semi_major_axis']]
X_cluster = np.log10(X_reduced.dropna())
X_cluster

# %%
# We then use the Kmeans algorithm to perform the classification.

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(X_cluster)
labels = kmeans.labels_

# %%
# Let's visualise what the clustering looks like as a function of pairs of
# physical parameters.

fig, ax = plt.subplots(1, 3, figsize=(15, 3))
ax[0].scatter(10 ** X_cluster['orbital_period'], 10 ** X_cluster['mass'], c=kmeans.labels_, cmap='viridis', s=20, alpha=0.6)
ax[1].scatter(10 ** X_cluster['orbital_period'], 10 ** X_cluster['radius'], c=kmeans.labels_, cmap='viridis', s=20, alpha=0.6)
ax[2].scatter(10 ** X_cluster['semi_major_axis'], 10 ** X_cluster['mass'], c=kmeans.labels_, cmap='viridis', s=20, alpha=0.6)
ax[1].set_xlim(0.1, 100.0)
ax[1].set_ylim(0.1, 10.0)
ax[2].set_xlim(0.001, 10000.0)
ax[2].set_ylim(0.0001, 100.0)
ax[0].set_xlabel('orbital period (day)')
ax[1].set_xlabel('orbital period (day)')
ax[2].set_xlabel('semimajor axis (au)')
ax[0].set_ylabel('mass')
ax[1].set_ylabel('radius')
ax[2].set_ylabel('mass')
fig.suptitle('Unsupervised Classification of Detected Expolanets')
[ax[i].set_xscale('log') for i in range(len(ax))]
[ax[i].set_yscale('log') for i in range(len(ax))]

plt.show()
# %%
# References
# ----------------------------------------
# arxiv: 2404.10522, 2210.14187
