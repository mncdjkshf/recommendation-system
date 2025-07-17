import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the items data
items = pd.read_csv("../app/data/items.csv")
items.head()

# Fill NaNs with empty strings
items['description'] = items['description'].fillna('')

# Optional: combine tags and description
if 'tags' in items.columns:
    items['combined'] = items['description'] + " " + items['tags']
else:
    items['combined'] = items['description']

tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform
tfidf_matrix = tfidf.fit_transform(items['combined'])

print("TF-IDF Matrix shape:", tfidf_matrix.shape)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Cosine Similarity Matrix shape:", cosine_sim.shape)

# Index mapping
indices = pd.Series(items.index, index=items['item_id']).drop_duplicates()

def get_similar_items(item_id, top_n=5):
    idx = indices[item_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    item_indices = [i[0] for i in sim_scores]
    return items.iloc[item_indices][['item_id', 'title', 'description']]

get_similar_items(10, top_n=5)

def plot_similarities(item_id):
    idx = indices[item_id]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    labels = items.iloc[[i[0] for i in sim_scores]]['title']
    scores = [i[1] for i in sim_scores]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=scores, y=labels)
    plt.title(f"Top 10 similar items to ID {item_id}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Item")
    plt.show()

plot_similarities(10)
