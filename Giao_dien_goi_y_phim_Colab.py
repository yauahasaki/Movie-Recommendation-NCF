# Chay cell nay SAU KHI chay xong Movie_Recommendation_Deep_Learning.py
# Can cac bien: model, ratings, movies, user_to_idx, movie_to_idx
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

required = ['model', 'ratings', 'movies', 'user_to_idx', 'movie_to_idx']
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError('Thieu bien: ' + ', '.join(missing) + '. Hay chay code huan luyen trong cung Notebook truoc.')

def goi_y_phim(user_id, top_k=10):
    user_id = int(user_id)
    if user_id not in user_to_idx:
        raise ValueError(f'User ID {user_id} khong co trong tap train. Thu ID khac.')
    da_xem = set(ratings.loc[ratings['userId'] == user_id, 'movieId'])
    ung_vien = [mid for mid in movie_to_idx if mid not in da_xem]
    if not ung_vien:
        return pd.DataFrame(columns=['movieId', 'title', 'genres', 'predicted_rating'])
    inputs = {
        'user': np.full(len(ung_vien), user_to_idx[user_id], dtype=np.int32),
        'movie': np.array([movie_to_idx[mid] for mid in ung_vien], dtype=np.int32),
    }
    scores = model.predict(inputs, batch_size=1024, verbose=0).reshape(-1)
    result = pd.DataFrame({'movieId': ung_vien, 'predicted_rating': scores})
    result = result.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    return (result.sort_values('predicted_rating', ascending=False)
                  .head(top_k).reset_index(drop=True))

user_input = widgets.BoundedIntText(value=1, min=1, max=int(max(user_to_idx)), description='User ID:')
button = widgets.Button(description='Gợi ý 10 phim', button_style='success')
output = widgets.Output()

def on_click(_):
    with output:
        clear_output(wait=True)
        try:
            result = goi_y_phim(user_input.value)
            print(f'TOP 10 PHIM GOI Y CHO USER {user_input.value}')
            if result.empty:
                print('Khong con phim phu hop trong danh sach ung vien.')
            else:
                display(result.rename(columns={'title': 'Ten phim', 'genres': 'The loai',
                                               'predicted_rating': 'Diem du doan', 'movieId': 'Ma phim'})
                              .round({'Diem du doan': 2}))
                result.to_csv('top10_goi_y_user_' + str(user_input.value) + '.csv', index=False)
                print('Da luu file CSV ket qua.')
        except Exception as exc:
            print('Loi:', exc)

button.on_click(on_click)
display(widgets.VBox([widgets.HTML('<h3>He thong goi y phim - Deep Learning NCF</h3>'),
                      widgets.HBox([user_input, button]), output]))
