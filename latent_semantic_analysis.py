from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
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

df = pd.read_csv('data/clean_genai-education_2023-2025.csv') 

print(df.describe())

# load dataset
df = df['text_clean'].dropna().tolist()

# Convert text to TF-IDF representation
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df)

# Apply LSA using Truncated SVD
lsa = TruncatedSVD(n_components=5, random_state=42)
X_lsa = lsa.fit_transform(X)

# Extract and display topics
terms = vectorizer.get_feature_names_out()
num_words = 10 # top words per topic

# evaluate model
print("\nExplained variance ratio (per topic):")
print(lsa.explained_variance_ratio_)
print("Total explained variance:", lsa.explained_variance_ratio_.sum())

# topic word distribution charts

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Top Words per Topic (LSA)', fontsize=16, fontweight='bold')

for i, comp in enumerate(lsa.components_):
    if i >= 5:  # only show first 5 topics
        break
    
    # get top words and their scores
    word_scores = [(terms[idx], comp[idx]) for idx in comp.argsort()[-num_words:]]
    word_scores.sort(key=lambda x: x[1], reverse=True)
    
    words, scores = zip(*word_scores)
    
    # create subplot
    ax = axes[i//3, i%3]
    bars = ax.barh(range(len(words)), scores, color=plt.cm.Set3(i))
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Topic {i+1}', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # add score labels on bars
    for j, (word, score) in enumerate(word_scores):
        ax.text(score + 0.001, j, f'{score:.3f}', 
                va='center', ha='left', fontsize=9)

# remove empty subplot
if len(lsa.components_) < 6:
    fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# explained variance visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# bar chart of explained variance
topic_labels = [f'Topic {i+1}' for i in range(len(lsa.explained_variance_ratio_))]
bars1 = ax1.bar(topic_labels, lsa.explained_variance_ratio_, 
                color=plt.cm.viridis(np.linspace(0, 1, len(lsa.explained_variance_ratio_))))
ax1.set_xlabel('Topics')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Explained Variance by Topic', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# add value labels on bars
for bar, ratio in zip(bars1, lsa.explained_variance_ratio_):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')

# pie chart of explained variance
colors = plt.cm.Set3(np.linspace(0, 1, len(lsa.explained_variance_ratio_)))
wedges, texts, autotexts = ax2.pie(lsa.explained_variance_ratio_, 
                                   labels=topic_labels,
                                   autopct='%1.1f%%',
                                   colors=colors,
                                   startangle=90)
ax2.set_title('Proportion of Variance Explained', fontweight='bold')

# make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()

print(f"Total variance explained: {lsa.explained_variance_ratio_.sum():.1%}")

# create summary statistics
summary_data = []
for i in range(lsa.n_components):
    top_words = [terms[idx] for idx in lsa.components_[i].argsort()[-10:]][::-1]
    summary_data.append({
        'Topic': f'Topic {i+1}',
        'Explained Variance': f'{lsa.explained_variance_ratio_[i]:.1%}',
        'Top 10 Words': ', '.join(top_words),
        'Avg Document Score': f'{X_lsa[:, i].mean():.3f}',
        'Score Std Dev': f'{X_lsa[:, i].std():.3f}'
    })

summary_df = pd.DataFrame(summary_data)
print("\nLSA TOPIC SUMMARY:")
print("="*170)
print(summary_df.to_string(index=False))

print("\n" + "="*60)
print(f"Generated {len(lsa.components_)} topics explaining {lsa.explained_variance_ratio_.sum():.1%} of total variance")
print("All visualizations have been displayed.")
print("="*60)