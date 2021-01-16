import json
from datetime import datetime
from random import randint, uniform

import numpy as np
import pandas as pd

# Probability dictionary
from matcher.clustering.utils import calculate_age

hobbies = ['Viết blog',
           'Ăn uống lành mạnh',
           'Mua sắm',
           'Sáng tạo nội dung',
           'Đi bộ đường dài hoặc Leo núi',
           'Vẽ tranh, Điêu khắc',
           'Chơi trò chơi điện tử',
           'Nấu ăn',
           'Đọc sách, báo, tiểu thuyết',
           'Xem phim']

rate = [0.16,
        0.16,
        0.01,
        0.19,
        0.11,
        0.05,
        0.10,
        0.09,
        0.07,
        0.06]


def gathering_profile_data():
    with open(r'../profiles.json') as json_file:
        data = json.load(json_file)
        list_user_id = []
        list_nick_name = []
        list_intro = []
        list_age = []
        list_gender = []
        list_height = []
        list_y = []
        list_x = []
        list_smoking = []
        list_drinking = []
        list_religious = []
        list_your_kids = []
        for user in data:
            list_user_id.append(user['user']['_id'])
            list_intro.append(user['user']['bio'])
            list_gender.append(user['user']['gender'])
            list_nick_name.append(user['user']['name'])

            date_time_str = user['user']['birth_date']
            date_time_obj = datetime.strptime(date_time_str[:10], '%Y-%m-%d')
            list_age.append(calculate_age(date_time_obj))

            list_height.append(randint(140, 200))
            list_x.append(uniform(106.478920, 106.837420))
            list_y.append(uniform(10.704097, 11.154205))
            list_smoking.append(randint(1, 3))
            list_drinking.append(randint(1, 3))
            list_religious.append(randint(1, 6))
            list_your_kids.append(randint(1, 3))

    final_df = pd.DataFrame(
        {'id': list_user_id, 'nickname': list_nick_name, 'introduction': list_intro, 'age': list_age,
         'gender': list_gender, 'height': list_height, 'x': list_x, 'y': list_y, 'smoking': list_smoking,
         'drinking': list_drinking, 'yourKids': list_your_kids, 'religious': list_religious})

    final_df['hobbies'] = list(np.random.choice(hobbies, size=(final_df.shape[0], 1, randint(1, 5)), p=rate))
    final_df['hobbies'] = final_df['hobbies'].apply(lambda x: list(set(x[0].tolist())))
    return final_df


def generate_new_profile(df):
    new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1] + 1])
    new_profile['id'] = "test"
    new_profile['nickname'] = "test"
    new_profile['introduction'] = "Vui vẻ không quạo"
    new_profile['age'] = randint(18, 60)
    new_profile['gender'] = randint(0, 1)
    new_profile['height'] = randint(140, 200)
    new_profile['x'] = uniform(106.478920, 106.837420)
    new_profile['y'] = uniform(10.704097, 11.154205)
    new_profile['smoking'] = randint(1, 3)
    new_profile['drinking'] = randint(1, 3)
    new_profile['yourKids'] = randint(1, 3)
    new_profile['religious'] = randint(1, 3)
    new_profile['hobbies'] = list(np.random.choice(hobbies, size=(1, randint(1, 5)), p=rate))
    new_profile['hobbies'] = new_profile['hobbies'].apply(lambda x: list(set(x.tolist())))
    return new_profile


def create_new_profile(df, id, nickname, introduction, age, gender, height, x, y, smoking, drinking,
                       your_kids, religious, list_hobbies):
    new_profile = pd.DataFrame(columns=df.columns, index=[df.index[-1] + 1])
    new_profile['id'] = id + nickname
    new_profile['nickname'] = nickname
    new_profile['introduction'] = introduction
    new_profile['age'] = age
    new_profile['gender'] = gender
    new_profile['height'] = height
    new_profile['x'] = x
    new_profile['y'] = y
    new_profile['smoking'] = smoking
    new_profile['drinking'] = drinking
    new_profile['yourKids'] = your_kids
    new_profile['religious'] = religious
    new_profile['hobbies'] = [list_hobbies]
    # new_profile['hobbies'] = new_profile['hobbies'].apply(lambda x: list(set(x.tolist())))
    return new_profile
