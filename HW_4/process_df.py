import pandas as pd
import numpy as np


def process(df,
            col_int=['is_churned', 'level',
                     'days_between_reg_fl',
                     'days_between_fl_df',
                     'has_return_date',
                     'has_phone_number'
                     ],
            col_float=['pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4', 'avg_min_ping_1', 'avg_min_ping_2',
                       'avg_min_ping_3', 'avg_min_ping_4', 'session_player_1', 'session_player_2', 'session_player_3',
                       'session_player_4', 'win_rate_1', 'win_rate_2', 'win_rate_3', 'win_rate_4', 'kd_1', 'kd_2',
                       'kd_3',
                       'kd_4', 'leavings_rate_1', 'leavings_rate_2', 'leavings_rate_3', 'leavings_rate_4',
                       'gold_spent_1',
                       'gold_spent_2', 'gold_spent_3', 'gold_spent_4', 'silver_spent_1', 'silver_spent_2',
                       'silver_spent_3', 'silver_spent_4', 'trans_amt_1', 'trans_amt_2', 'trans_amt_3', 'trans_amt_4',
                       'disconnect_amt_1', 'disconnect_amt_2', 'disconnect_amt_3', 'disconnect_amt_4',
                       'session_amt_1', 'session_amt_2', 'session_amt_3', 'session_amt_4',
                       'reports_amt_1', 'reports_amt_2', 'reports_amt_3', 'reports_amt_4',
                       'sess_with_abusers_amt_1', 'sess_with_abusers_amt_2', 'sess_with_abusers_amt_3',
                       'sess_with_abusers_amt_4',
                       'age', 'gender']):
    print('Drop dublicates...')
    df.drop_duplicates(inplace=True)
    for index_row in df.index[df['user_id'] == 'user_id'].tolist():
        df.drop([index_row], inplace=True)
    # удаляем логины пользователей, они нам не нужны для построения модели
    print('Drop user ID ...')
    df.drop(columns=['user_id'], inplace=True)
    print('Process gender and age...')
    df.loc[df['gender'] == 'M', 'gender'] = 0
    df.loc[df['gender'] == 'F', 'gender'] = 1
    print('Canges type ...')
    for column in col_int:
        df[column] = df[column].astype(int)
    for column in col_float:
        df[column] = df[column].astype(float)
    # возраст и пол заполняем модой
    mode_age = int(float(df['age'].mode()[0]))
    mode_gender = int(float(df['gender'].mode()[0]))
    df['age'].fillna(mode_age, inplace=True)  # 19
    df['gender'].fillna(mode_gender, inplace=True)  # 0
    # для всех остальных признаков очень маловероятно, что какие-то данные потерялись,
    # следовательно если их нет, то игрок в этот день не заходил в игру или не совершал транзакций, заполняем 0
    print('Process NULL values ...')
    df.fillna(0, inplace=True)
    print('Another process ...')
    df.loc[df['gender'] < 0, 'gender'] = df.loc[df['gender'] < 0, 'gender'] * -1
    df.loc[df['age'] < 7, 'age'] = mode_age
    df.loc[df['age'] > 70, 'age'] = mode_age
    df.loc[df['avg_min_ping_1'] < 0, 'avg_min_ping_1'] = 0
    df.loc[df['avg_min_ping_2'] < 0, 'avg_min_ping_2'] = 0
    df.loc[df['avg_min_ping_3'] < 0, 'avg_min_ping_3'] = 0
    df.loc[df['avg_min_ping_4'] < 0, 'avg_min_ping_4'] = 0
    for param in [
        'trans_amt_1', 'trans_amt_2', 'trans_amt_3', 'trans_amt_4',
        'pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4',
        'reports_amt_1', 'reports_amt_2', 'reports_amt_3', 'reports_amt_4',
        'sess_with_abusers_amt_1', 'sess_with_abusers_amt_2', 'sess_with_abusers_amt_3', 'sess_with_abusers_amt_4',
        'session_amt_1', 'session_amt_2', 'session_amt_3', 'session_amt_4',
        'disconnect_amt_1', 'disconnect_amt_2', 'disconnect_amt_3', 'disconnect_amt_4',
        'avg_min_ping_1', 'avg_min_ping_2', 'avg_min_ping_3', 'avg_min_ping_4',
        'session_player_1', 'session_player_2', 'session_player_3', 'session_player_4',
        'gold_spent_1', 'gold_spent_2', 'gold_spent_3', 'gold_spent_4',
        'silver_spent_1', 'silver_spent_2', 'silver_spent_3', 'silver_spent_4'

    ]:
        df[param] = df[param].round(0)
        df[param] = df[param].astype(int)
    return df
