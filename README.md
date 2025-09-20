# Generative AI in Education

**Group Name:** *KaizenTek*

**Members:** 
- BRIONES, Neil Angelo
- CASILEN, Kurshan Craig Sandler
- FORTALEZA, Keanu Sonn
- LACTAOTAO, Benny Gil
- MORENO, Jun Laurenz
- RAMOS, Albert Jannsen
- SALON, Ruzzlee

## Scope

End-to-end text analytics on **Generative AI in Education**:

* **Topic Modeling:** LDA, LSA, BERTopic (≥3 topics identified)
* **Sentiment Analysis:** Supervised classifier on the same dataset

Dataset: **≥200 public text records** (social media, blogs, news). Basic cleaning, de-duplication, and preprocessing applied.


## Proccess

1. **Collect & Prepare Data:** gather public texts on the topic; clean (URLs, punctuation), tokenize, lemmatize, remove stopwords.
2. **Explore:** basic stats (counts, lengths, sources) to confirm topical variety.
3. **Topic Modeling:** train LDA, LSA, BERTopic; pick **K** topics; label topics.
4. **Evaluate Topics:** coherence & diversity + quick interpretability check.
5. **Sentiment Pipeline:**

   * Filter opinionated records.
   * **Splits:** 10% **unseen validation**; remaining 90% → **80/20 train/test**.
   * Train baseline classifier (e.g., Logistic Regression/SVM on TF-IDF).
   * Report accuracy, precision/recall/F1, confusion matrix.
6. **Iterate:** light tuning (topic count, vectorization; class weights/regularization).
7. **Discuss & Conclude:** summarize key topics and sentiment findings specific to **Generative AI in Education**