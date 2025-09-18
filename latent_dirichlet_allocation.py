from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# load dataset
df = pd.read_csv('data/clean_genai-education_2023-2025.csv')
df = df['text_clean'].dropna().tolist()

# convert test to bag of words representation
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df)

# apply LDA
lda = LatentDirichletAllocation(n_components=5, random_state=42)
X_lda = lda.fit_transform(X)

# extract and display topics
terms = vectorizer.get_feature_names_out()
num_words = 10 # top words per topic

# evaluate model
print("\nPerplexity (lower is better):", lda.perplexity(X))

# log-likelihood score (higher is better)
print("Log-likelihood:", lda.score(X))

# calculate topic proportions for visualization
# using the average document-topic distribution as proxy for topic importance
topic_proportions = X_lda.mean(axis=0)
topic_proportions = topic_proportions / topic_proportions.sum()  # normalize to sum to 1


# topic word distribution charts
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Top Words per Topic (LDA)', fontsize=16, fontweight='bold')

for i, topic in enumerate(lda.components_):
    if i >= 5:  # only show first 5 topics
        break
    
    # Get top words and their scores
    word_indices = topic.argsort()[-num_words:]
    word_scores = [(terms[idx], topic[idx]) for idx in word_indices]
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    words, scores = zip(*word_scores)
    
    # create subplot
    ax = axes[i//3, i%3]
    bars = ax.barh(range(len(words)), scores, color=plt.cm.Set3(i))
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('Probability Score')
    ax.set_title(f'Topic {i+1}', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # add score labels on bars
    for j, (word, score) in enumerate(word_scores):
        ax.text(score + max(scores)*0.01, j, f'{score:.3f}', 
                va='center', ha='left', fontsize=9)

# remove empty subplot
if len(lda.components_) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# explained variance visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# bar chart of topic proportions
topic_labels = [f'Topic {i+1}' for i in range(len(topic_proportions))]
bars1 = ax1.bar(topic_labels, topic_proportions, 
                color=plt.cm.viridis(np.linspace(0, 1, len(topic_proportions))))
ax1.set_xlabel('Topics')
ax1.set_ylabel('Average Topic Proportion')
ax1.set_title('Topic Importance (Average Document Distribution)', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# add value labels on bars
for bar, prop in zip(bars1, topic_proportions):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + max(topic_proportions)*0.01,
             f'{prop:.3f}', ha='center', va='bottom', fontweight='bold')

# pie chart of topic proportions
colors = plt.cm.Set3(np.linspace(0, 1, len(topic_proportions)))
wedges, texts, autotexts = ax2.pie(topic_proportions, 
                                   labels=topic_labels,
                                   autopct='%1.1f%%',
                                   colors=colors,
                                   startangle=90)
ax2.set_title('Proportional Topic Distribution', fontweight='bold')

# make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()

print(f"Most prominent topic: Topic {np.argmax(topic_proportions)+1} ({topic_proportions.max():.1%})")
print(f"Least prominent topic: Topic {np.argmin(topic_proportions)+1} ({topic_proportions.min():.1%})")

# calculate topic statistics
summary_data = []
for i in range(lda.n_components):
    top_words = [terms[idx] for idx in lda.components_[i].argsort()[-10:]][::-1]
    summary_data.append({
        'Topic': f'Topic {i+1}',
        'Avg Proportion': f'{topic_proportions[i]:.1%}',
        'Top 10 Words': ', '.join(top_words),
        'Avg Doc Probability': f'{X_lda[:, i].mean():.3f}',
        'Std Dev': f'{X_lda[:, i].std():.3f}'
    })

summary_df = pd.DataFrame(summary_data)
print("\nLDA TOPIC SUMMARY:")
print("="*170)
print(summary_df.to_string(index=False))

# performance summary
print("\nModel Performance Summary:")
print("="*50)
print(f"Number of Topics: {lda.n_components}")
print(f"Perplexity (lower is better): {lda.perplexity(X):.2f}")
print(f"Log-likelihood (higher is better): {lda.score(X):.2f}")
print(f"Most balanced topic distribution: {1-max(topic_proportions)/min(topic_proportions):.2f}")