import pandas as 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)

positive_tweets = [
    "I absolutely love this product! Best purchase ever!",
    "Amazing experience, highly recommend to everyone!",
    "This brand never disappoints. Fantastic quality!",
    "So happy with my order. Great customer service too!",
    "Incredible value for money. Will buy again!",
    "Best app I've used. Makes life so much easier!",
    "Outstanding performance, exceeded all expectations!",
    "Wonderful product, my whole family loves it!",
    "Five stars! Shipping was fast and product is perfect.",
    "Loving every feature of this. Totally worth it!",
    "Great quality, exactly as described. Very satisfied!",
    "Superb design and functionality. Highly impressed!",
    "This exceeded my expectations in every way possible.",
    "Brilliant! Works perfectly and looks amazing too.",
    "Very happy customer here. Will definitely recommend."
]

negative_tweets = [
    "Worst product I ever bought. Completely disappointed!",
    "Terrible customer service. Never buying from them again.",
    "This is a waste of money. Broke after one week!",
    "Very unhappy with my purchase. Quality is awful.",
    "Do NOT buy this. It's a total scam and fraud.",
    "Horrible experience. Delayed shipping and broken item.",
    "Completely useless product. Requesting a refund now.",
    "So frustrated with this brand. Nothing works properly.",
    "Defective product and no response from support team.",
    "This is the worst thing I have ever purchased. Garbage!",
    "Total disappointment. Does not match the description at all.",
    "Broke immediately. Cheap materials and poor construction.",
    "The worst customer experience I've had. Avoid this brand!",
    "Returning this immediately. Completely unusable product.",
    "Overpriced garbage. Absolutely nothing works as advertised."
]

neutral_tweets = [
    "Just received my order. Will update after testing it.",
    "The product is okay. Neither great nor terrible I guess.",
    "Delivered on time. Haven't tried it fully yet.",
    "Looks decent in person. Matches the photos online.",
    "Normal product, does what it says on the box.",
    "Got the item. Packaging was simple and straightforward.",
    "Product arrived. Will write a detailed review later.",
    "Average quality for the price. Nothing special.",
    "It works as expected. No major complaints so far.",
    "Received the package yesterday. Seems fine overall."
]

tweets = positive_tweets + negative_tweets + neutral_tweets
topics = (['Brand Review'] * 15 + ['Product Feedback'] * 15 + ['Delivery'] * 10)
np.random.shuffle(tweets)

df = pd.DataFrame({'text': tweets, 'topic': np.random.choice(
    ['Brand Review', 'Product Feedback', 'Delivery', 'Customer Service'], len(tweets))})


print("=" * 50)
print("SENTIMENT ANALYSIS RESULTS")
print("=" * 50)

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive', compound
    elif compound <= -0.05:
        return 'Negative', compound
    else:
        return 'Neutral', compound

df[['vader_sentiment', 'vader_score']] = df['text'].apply(
    lambda x: pd.Series(get_vader_sentiment(x)))

# TextBlob sentiment
def get_textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        return 'Positive'
    elif polarity < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['textblob_sentiment'] = df['text'].apply(get_textblob_sentiment)

print(f"\nVADER Sentiment Distribution:\n{df['vader_sentiment'].value_counts()}")
print(f"\nTextBlob Sentiment Distribution:\n{df['textblob_sentiment'].value_counts()}")
print(f"\nSample Results:\n{df[['text','vader_sentiment','vader_score']].head(8)}")


fig = plt.figure(figsize=(18, 12))
fig.suptitle('Task 04 – Social Media Sentiment Analysis', fontsize=16, fontweight='bold')

color_map = {'Positive': '#2ECC71', 'Negative': '#E74C3C', 'Neutral': '#95A5A6'}

ax1 = fig.add_subplot(2, 3, 1)
sentiment_counts = df['vader_sentiment'].value_counts()
bars = ax1.bar(sentiment_counts.index,
               sentiment_counts.values,
               color=[color_map[s] for s in sentiment_counts.index],
               edgecolor='white', linewidth=1.2)
ax1.set_title('VADER Sentiment Distribution', fontweight='bold')
ax1.set_ylabel('Count')
for bar, val in zip(bars, sentiment_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, val+0.3, str(val),
             ha='center', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

ax2 = fig.add_subplot(2, 3, 2)
tb_counts = df['textblob_sentiment'].value_counts()
ax2.bar(tb_counts.index, tb_counts.values,
        color=[color_map[s] for s in tb_counts.index],
        edgecolor='white', linewidth=1.2)
ax2.set_title('TextBlob Sentiment Distribution', fontweight='bold')
ax2.set_ylabel('Count')
for i, val in enumerate(tb_counts.values):
    ax2.text(i, val+0.3, str(val), ha='center', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)


ax3 = fig.add_subplot(2, 3, 3)
colors_hist = [color_map[s] for s in df['vader_sentiment']]
ax3.hist(df['vader_score'], bins=20, color='#4C72B0', edgecolor='white', alpha=0.85)
ax3.axvline(0.05, color='green', linestyle='--', label='Positive threshold')
ax3.axvline(-0.05, color='red', linestyle='--', label='Negative threshold')
ax3.set_title('VADER Compound Score Distribution', fontweight='bold')
ax3.set_xlabel('Compound Score')
ax3.set_ylabel('Frequency')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)


ax4 = fig.add_subplot(2, 3, 4)
topic_sentiment = pd.crosstab(df['topic'], df['vader_sentiment'])
topic_sentiment.plot(kind='bar', ax=ax4,
                     color=[color_map.get(c, '#888') for c in topic_sentiment.columns],
                     edgecolor='white', rot=15)
ax4.set_title('Sentiment by Topic', fontweight='bold')
ax4.set_ylabel('Count')
ax4.legend(title='Sentiment', fontsize=9)
ax4.grid(axis='y', alpha=0.3)


ax5 = fig.add_subplot(2, 3, 5)
positive_text = ' '.join(df[df['vader_sentiment'] == 'Positive']['text'])
if positive_text.strip():
    wc_pos = WordCloud(width=400, height=250, background_color='white',
                       colormap='Greens', max_words=50).generate(positive_text)
    ax5.imshow(wc_pos, interpolation='bilinear')
ax5.axis('off')
ax5.set_title('Positive Sentiment WordCloud', fontweight='bold')

ax6 = fig.add_subplot(2, 3, 6)
negative_text = ' '.join(df[df['vader_sentiment'] == 'Negative']['text'])
if negative_text.strip():
    wc_neg = WordCloud(width=400, height=250, background_color='white',
                       colormap='Reds', max_words=50).generate(negative_text)
    ax6.imshow(wc_neg, interpolation='bilinear')
ax6.axis('off')
ax6.set_title('Negative Sentiment WordCloud', fontweight='bold')

plt.tight_layout()
plt.savefig('task04_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nTask 04 complete! Output saved as task04_output.png")
