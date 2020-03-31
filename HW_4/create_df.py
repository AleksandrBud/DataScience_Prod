import time
import pandas as pd
from datetime import datetime, timedelta


def time_format(sec):
    return str(timedelta(seconds=sec))


def build_dataset_raw(churned_start_date='2019-01-01',
                      churned_end_date='2019-02-01',
                      inter_list=[(1, 7), (8, 14)],
                      raw_data_path='train/',
                      dataset_path='dataset/',
                      mode='train',
                      csv_file_list=['payments', 'reports', 'abusers', 'logins', 'pings', 'sessions', 'shop']):
    start_t = time.time()

    samples = pd.read_csv('{}sample.csv'.format(raw_data_path),
                          sep=';',
                          chunksize=10000,
                          na_values=['\\N', 'None'],
                          encoding='utf-8')

    print('Run time (reading csv files): {}'.format(time_format(time.time() - start_t)))
    # -----------------------------------------------------------------------------------------------------
    print('NO dealing with outliers, missing values and categorical features...')
    # -----------------------------------------------------------------------------------------------------
    # На основании дня отвала (last_login_dt) строим признаки, которые описывают активность игрока перед уходом

    print('Creating dataset...')
    # Для каждой подвыборки пользователей будем создавать признаки и записывать в финальный DataSet
    for sample in samples:
        # Создадим пустой датасет - в зависимости от режима построения датасета - train или test
        if mode == 'train':
            dataset = sample.copy()[['user_id', 'is_churned', 'level', 'donate_total']]
        elif mode == 'test':
            dataset = sample.copy()[['user_id', 'level', 'donate_total']]

        # Пройдемся по всем источникам, содержащим "динамичекие" данные
        for file_name in csv_file_list:
            df = pd.read_csv('{}.csv'.format(raw_data_path + file_name), sep=';', na_values=['\\N', 'None'],
                             encoding='utf-8')
            # Получим 'day_num_before_churn' для каждого из значений в источнике для определения недели
            data = pd.merge(sample[['user_id', 'login_last_dt']], df, on='user_id')
            data['day_num_before_churn'] = 1 + (
                    data['login_last_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) -
                    data['log_dt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))).apply(lambda x: x.days)
            df_features = data[['user_id']].drop_duplicates().reset_index(drop=True)

            # Для каждого признака создадим признаки для каждого из времененно интервала (в нашем примере 4 интервала
            # по 7 дней)
            features = list(set(data.columns) - set(['user_id', 'login_last_dt', 'log_dt', 'day_num_before_churn']))
            print('Processing with features:', features)
            for feature in features:
                for i, inter in enumerate(inter_list):
                    inter_df = data.loc[data['day_num_before_churn'].between(inter[0], inter[1], inclusive=True)]. \
                        groupby('user_id')[feature].mean().reset_index(). \
                        rename(index=str, columns={feature: feature + '_{}'.format(i + 1)})
                    df_features = pd.merge(df_features, inter_df, how='left', on='user_id')

            # Добавляем построенные признаки в датасет
            dataset = pd.merge(dataset, df_features, how='left', on='user_id')

            print('Run time (calculating features): {}'.format(time_format(time.time() - start_t)))

        # Добавляем "статические" признаки
        profiles = pd.read_csv('{}profiles.csv'.format(raw_data_path), sep=';', na_values=['\\N', 'None'],
                               encoding='utf-8')
        dataset = pd.merge(dataset, profiles, how='left', on='user_id')
        # ---------------------------------------------------------------------------------------------------------------------------
        dataset.to_csv('{}dataset_raw_{}.csv'.format(dataset_path, mode), sep=';', mode='a', index=False)
        print('Dataset is successfully built and saved to {}, run time "build_dataset_raw": {}'. \
              format(dataset_path, time_format(time.time() - start_t)))
    return pd.read_csv('{}dataset_raw_{}.csv'.format(dataset_path, mode), sep=';', encoding='utf-8')
