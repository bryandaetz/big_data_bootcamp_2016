### Big Data Bootcamp 2016: Text Modeling with R & Python
#### Frank D. Evans, Exaptive


##### Description
This is a workshop on how to build multiple kinds of text models in both the R and Python domain. It will concentrate on models such as LDA Topic Modeling and Sentiment Analysis, as well as the methods that can take these models to the next level. The workshop will additionally walk through how to effectively prep text data for text modeling, and wrangle the text to get a clean model. This is a hands on workshop, not just a lecture. We will walk through live coding of the entire process, testing and refining our code as we go.

This is a run down of everything you need to have in place to attend the workshop. Make sure you can download the database and have all of the requisite libraries installed in Python, R, or both depending on which aspects of the workshop you would like to work through live.

Additionally, you can clone this repository which will download the data as well as ensure you can later pull to get the full version of the code as built during the session.

##### Data
The data set we will be using is the text of the President's State of the Union address. There is a file in this repo called `sotu_parsed.json`. Ensure that you can either download this file and have it available to your local system, or that you clone the repo and have the data available.


##### Python Domain
_*Libraries Needed*_  
Ensure that you can make these imports from your local Python interpreter. All packages are either part of the standard library or are pip installable.

```
# Standard Library
import json
import re

# Pip installable
import numpy
import nltk
import sklearn
import lda

# Ensure the WordNet asset is loaded
from nltk.stem import WordNetLemmatizer
assert WordNetLemmatizer().lemmatize('dogs') == 'dog'

# If not installed:
nltk.download('wordnet')
```

_*Environment*_  
For this demonstration, the latest version of Python 2.7 will be used. I will use the Atom text editor, and the mac bash shell on screen for the demo. There is nothing special I will use for either, you are welcome to use any editor you like and should have no problem following along. You can additionally use an IDE like Rodeo, Jupyter, Spyder if you prefer--and should have no issue following along with the content.


##### R Domain
_*Libraries Needed*_  
Ensure that you can make these library loads from your R shell or IDE. All packages are CRAN installable.
```
library(jsonlite)
library(stringr)
library(dplyr)
library(tm)
library(topicmodels)
library(sentiment)
```

_*Environment*_  
For this demonstration, R version 3.3.x will be used. I will use the RStudio development environment for the demo. You are welcome to do the same, as you are completely able to use another method such as text editor/bash shell, Jupyter, or any other workflow you are familiar with, and should have no issue following along with the content.
