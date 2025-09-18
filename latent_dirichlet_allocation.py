from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np

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

for i, topic in enumerate(lda.components_):
    terms_in_topic = [terms[x] for x in topic.argsort()[-num_words:]]
    print(f"\nTopic {i+1}:")
    print(", ".join(terms_in_topic))

# evaluate model
print("\nPerplexity (lower is better):", lda.perplexity(X))

# Log-likelihood score (higher is better)
print("Log-likelihood:", lda.score(X))