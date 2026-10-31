# Recommendation Systems for Movies - Neural Collaborative Filtering
# Run in Google Colab or local Python with tensorflow, pandas, numpy, sklearn, matplotlib
import os, zipfile, urllib.request, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, Model

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
if not os.path.exists('ml-latest-small/ratings.csv'):
    print('Downloading MovieLens Latest Small...')
    urllib.request.urlretrieve(URL, 'ml-latest-small.zip')
    with zipfile.ZipFile('ml-latest-small.zip') as z: z.extractall('.')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
print('Ratings:', len(ratings), 'Users:', ratings.userId.nunique(), 'Movies:', ratings.movieId.nunique())
print(ratings.rating.describe())
ratings.rating.value_counts().sort_index().plot(kind='bar', title='Rating distribution')
plt.xlabel('Rating'); plt.ylabel('Count'); plt.tight_layout(); plt.savefig('eda_ratings.png'); plt.show()

# Split interactions first, with fixed random seed. This is a random-interaction evaluation.
train_df, temp_df = train_test_split(ratings, test_size=0.30, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)
# Fit ID mappings only on train; unknown IDs in validation/test cannot be scored.
user_to_idx = {u:i for i,u in enumerate(sorted(train_df.userId.unique()))}
movie_to_idx = {m:i for i,m in enumerate(sorted(train_df.movieId.unique()))}
def encode(frame):
    frame = frame[frame.userId.isin(user_to_idx) & frame.movieId.isin(movie_to_idx)].copy()
    frame['u'] = frame.userId.map(user_to_idx).astype('int32')
    frame['m'] = frame.movieId.map(movie_to_idx).astype('int32')
    return frame
train, val, test = map(encode, [train_df, val_df, test_df])
print('Train/Val/Test:', len(train), len(val), len(test))

# Baseline: global mean rating (fit only on train)
baseline = float(train.rating.mean())
def evaluate(y, pred, name):
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    print(f'{name}: RMSE={rmse:.4f}, MAE={mae:.4f}')
    return {'Model':name, 'RMSE':rmse, 'MAE':mae}

# Neural Collaborative Filtering: user/movie embeddings + multilayer perceptron
n_users, n_movies = len(user_to_idx), len(movie_to_idx)
u_input = layers.Input(shape=(1,), dtype='int32', name='user')
m_input = layers.Input(shape=(1,), dtype='int32', name='movie')
u_emb = layers.Flatten()(layers.Embedding(n_users, 32, name='user_embedding')(u_input))
m_emb = layers.Flatten()(layers.Embedding(n_movies, 32, name='movie_embedding')(m_input))
x = layers.Concatenate()([u_emb, m_emb])
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
# Ratings in MovieLens are from 0.5 to 5.0
out = layers.Lambda(lambda z: 0.5 + 4.5 * z)(layers.Dense(1, activation='sigmoid')(x))
model = Model([u_input, m_input], out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()
def xy(df):
    return {'user':df.u.to_numpy(), 'movie':df.m.to_numpy()}, df.rating.to_numpy(dtype='float32')
X_train, y_train = xy(train)
X_val, y_val = xy(val)
X_test, y_test = xy(test)
history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=20,
                    batch_size=256, callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=3, restore_best_weights=True)], verbose=2)
# Evaluate test once after choosing model by validation.
pred = model.predict(X_test, verbose=0).ravel()
results = [evaluate(y_test, np.full_like(y_test, baseline), 'Global mean baseline'),
           evaluate(y_test, pred, 'Deep Learning NCF')]
pd.DataFrame(results).to_csv('evaluation_results.csv', index=False)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title('Learning curves'); plt.legend()
plt.tight_layout(); plt.savefig('training_loss.png'); plt.show()

# Top-10 movies not rated by this user in the full available history.
# For a real deployment, use only historical interactions available at recommendation time.
def recommend_top10(user_id):
    if user_id not in user_to_idx:
        raise ValueError('User ID not seen during training')
    watched = set(ratings.loc[ratings.userId == user_id, 'movieId'])
    candidates = [mid for mid in movie_to_idx if mid not in watched]
    if not candidates: return pd.DataFrame(columns=['title','predicted_rating'])
    u = np.full(len(candidates), user_to_idx[user_id], dtype='int32')
    m = np.array([movie_to_idx[mid] for mid in candidates], dtype='int32')
    scores = model.predict({'user':u, 'movie':m}, verbose=0).ravel()
    recs = pd.DataFrame({'movieId':candidates, 'predicted_rating':scores})
    recs = recs.merge(movies[['movieId','title']], on='movieId', how='left')
    return recs.sort_values('predicted_rating', ascending=False).head(10)[['title','predicted_rating']]

example_user = int(train.userId.iloc[0])
print('Top 10 recommendations for user:', example_user)
print(recommend_top10(example_user).to_string(index=False))
model.save('movie_ncf.keras')
print('Saved: movie_ncf.keras, evaluation_results.csv, training_loss.png, eda_ratings.png')
