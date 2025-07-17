import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
import joblib
import os

# Load the ratings data
ratings = pd.read_csv("../app/data/ratings.csv")
ratings.head()

# Define the format
reader = Reader(rating_scale=(ratings.rating.min(), ratings.rating.max()))

# Load dataset into Surprise format
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# Create trainset and testset
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Instantiate the model
model = SVD()

# Train the model
model.fit(trainset)

from surprise import accuracy

# Predict on test set
predictions = model.test(testset)

# Evaluate metrics
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

from surprise import accuracy

# Predict on test set
predictions = model.test(testset)

# Evaluate metrics
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# Evaluate model using 5-fold cross-validation
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

output_dir = "../app/models"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(model, os.path.join(output_dir, "collaborative_model.pkl"))
print("✅ Model saved.")

# Predict for a user and an item
user_id = "1"
item_id = "10"
prediction = model.predict(user_id, item_id)
print(f"Predicted rating of user {user_id} for item {item_id}: {prediction.est:.2f}")

# Get all items
all_items = ratings['item_id'].unique()


# Recommend top-N items not yet rated by user
def get_top_n(user_id, model, ratings_df, n=5):
    rated = ratings_df[ratings_df['user_id'] == int(user_id)]['item_id'].tolist()
    unrated = [iid for iid in all_items if iid not in rated]

    predictions = [model.predict(str(user_id), str(iid)) for iid in unrated]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    return [(pred.iid, pred.est) for pred in top_n]


# Example
top_recs = get_top_n("1", model, ratings, n=5)
print("Top-5 Recommendations:", top_recs)

# Get all items
all_items = ratings['item_id'].unique()


# Recommend top-N items not yet rated by user
def get_top_n(user_id, model, ratings_df, n=5):
    rated = ratings_df[ratings_df['user_id'] == int(user_id)]['item_id'].tolist()
    unrated = [iid for iid in all_items if iid not in rated]

    predictions = [model.predict(str(user_id), str(iid)) for iid in unrated]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    return [(pred.iid, pred.est) for pred in top_n]


# Example
top_recs = get_top_n("1", model, ratings, n=5)
print("Top-5 Recommendations:", top_recs)
