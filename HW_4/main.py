import pandas as pd
import create_df
import process_df
import FitModel
import PredictValue
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN


CHURNED_START_DATE = '2019-09-01'
CHURNED_END_DATE = '2019-10-01'
INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]
TRAIN_PATH = '../train/'
DATA_SET_PATH = './dataset/'
INT_COL = ['is_churned', 'level', 'days_between_reg_fl', 'days_between_fl_df', 'has_return_date', 'has_phone_number']
FLOAT_COL = ['pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4', 'avg_min_ping_1', 'avg_min_ping_2',
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
             'age', 'gender'
             ]
# Собираем Dataset
df = create_df.build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                                 churned_end_date=CHURNED_END_DATE,
                                 inter_list=INTER_LIST,
                                 raw_data_path=TRAIN_PATH,
                                 dataset_path=DATA_SET_PATH,
                                 mode='train')
df = pd.read_csv('{}dataset_raw_{}.csv'.format(DATA_SET_PATH, 'train'), sep=';', encoding='utf-8')
# Обрабатываем данные
df = process_df.process(df, INT_COL, FLOAT_COL)
TARGET_NAME = 'is_churned'
BASE_FEATURE_NAMES = df.columns.drop(TARGET_NAME).tolist()
X = df.drop([TARGET_NAME], axis=1)
y = df[TARGET_NAME]

# Нормализуем данные
X_mm = MinMaxScaler().fit_transform(df[BASE_FEATURE_NAMES])
# Разбиваем данные
print('Split data ...')
x_train, x_test, y_train, y_test = train_test_split(MinMaxScaler().fit_transform(df[BASE_FEATURE_NAMES]),  # X_mm
                                                    df[TARGET_NAME],  # y
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=df[TARGET_NAME],  # y
                                                    random_state=100)
# Делаем балансировку тренировочного набора
print('Balansing data ...')
smote_enn = SMOTEENN(random_state=42)
x_train_balanced, y_train_balanced = smote_enn.fit_resample(x_train, y_train)

x_train_balanced.columns = BASE_FEATURE_NAMES
x_test.columns = BASE_FEATURE_NAMES
print('Fit Model ...')
model = FitModel.CatBoostFit(x_train_balanced, y_train_balanced, x_test, y_test)
y_predict = PredictValue.Predict(model, x_test)
