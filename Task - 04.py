import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from textblob import TextBlob
from wordcloud import WordCloud

nltk.download('punkt')

sns.set_style("whitegrid")


df = pd.read_csv(
    "social.csv",
    header=None,
    names=["id", "topic", "label", "text"]
)

print(df.head())
print(df.info())

def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["Predicted_Sentiment"] = df["text"].apply(get_sentiment)

print(df.head())

sns.countplot(x="Predicted_Sentiment", data=df)
plt.title("Sentiment Distribution (Predicted)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

sentiment_counts = df["Predicted_Sentiment"].value_counts(normalize=True) * 100
print(sentiment_counts)

positive_text = " ".join(
    df[df["Predicted_Sentiment"] == "Positive"]["text"].dropna()
)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Sentiment WordCloud")
plt.show()
