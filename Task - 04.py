import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from textblob import TextBlob
from wordcloud import WordCloud

# Download required NLTK data
nltk.download('punkt')

# Set seaborn style
sns.set_style("whitegrid")

# =========================
# Load Dataset
# =========================
df = pd.read_csv("social.csv")
