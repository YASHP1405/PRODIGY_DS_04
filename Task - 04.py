import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from textblob import TextBlob
from wordcloud import WordCloud

nltk.download('punkt')

sns.set_style("whitegrid")

df = pd.read_csv("social.csv")

print(df.head())
print(df.info())

# =========================
# Sentiment Analysis Function
# =========================
def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df['Sentiment'] = df['text'].apply(get_sentiment)

print(df.head())

# =========================
# Sentiment Distribution Plot
# =========================
sns.countplot(x='Sentiment', data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# =========================
# Sentiment Percentage
# =========================
sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
print(sentiment_counts)

# =========================
# WordCloud for Positive Sentiment
# =========================
positive_text = " ".join(df[df['Sentiment'] == 'Positive']['text'].dropna())

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Sentiment WordCloud")
plt.show()