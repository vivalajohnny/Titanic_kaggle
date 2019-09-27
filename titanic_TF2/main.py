from data_manipulation import *
from sklearn.model_selection import train_test_split    # Для разделения изначального датасета
import tensorflow as tf
import os   # Для отключения ошибки
from tensorflow import feature_column   # Манипуляции с анализируемыми колонками
from tensorflow.keras import layers     # Создание слоев нейросети

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Отключение ошибки


# RangeIndex: 891 entries, 0 to 890
# Data columns (total 19 columns):
# PassengerId         891 non-null int64
# Survived            891 non-null int64
# Pclass              891 non-null int64
# Name                891 non-null object
# Sex                 891 non-null int64
# Age                 891 non-null int64
# SibSp               891 non-null int64
# Parch               891 non-null int64
# Ticket              891 non-null object
# Fare                891 non-null int64
# Cabin               891 non-null object
# Embarked            891 non-null int64
# Title               891 non-null object
# childs_w_parents    891 non-null int64
# parents_w_childs    891 non-null int64
# relatives           891 non-null int64
# single              891 non-null int64
# floor               891 non-null int64
# cabin_number        891 non-null int64
# dtypes: int64(15), object(4)
# memory usage: 132.4+ KB

# Преобразование исходных в датафреймов в нужный формат с помощью функции написанной заранее
train_df = data_manipulation(train_data)
test_df = data_manipulation(test_data)


# Список с анализируемыми фичами
l_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'childs_w_parents',
                'parents_w_childs', 'relatives', 'single', 'floor', 'cabin_number']

# Разделяем исходный тренировочный датасет на сет тренировки и сет валидации. TRAIN VAL TEST
train, val = train_test_split(train_df, test_size=0.2)

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    '''Функция преобразования pandas dataframe в dataset TensorFlow.
    Необходимо преобразовать лишь train dataframe и validation dataframe.'''
    dataframe = dataframe.copy()
    labels = dataframe.pop('Survived')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe[l_features]), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

# Преобразование тестового датафрейм в датасет
test_ds = tf.data.Dataset.from_tensor_slices(dict(test_df[l_features]))
test_ds = test_ds.batch(32)
# Преобразование трэйн и вал дф в датасет
train_ds = df_to_dataset(train)
val_ds = df_to_dataset(val)

print('TRAIN_ds============')
print(train_ds)
print('VAL_ds============')
print(val_ds)
print('TEST_ds============')
print(test_ds)


# Список зафичеренных колонок в специальном формате
feature_columns = []

# numeric cols
for header in l_features:
  feature_columns.append(feature_column.numeric_column(header))

'''Все необходимые манипуляции сделаны,
теперь необходимо создать архитектуру нейронной сети'''

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=30)

array_p = model.predict(test_ds)

total_alive_count = 160

df = pd.DataFrame(array_p)
df.rename(columns={0: 'Survived'}, inplace=True)
df['PassengerId'] = test_df['PassengerId']
df = df[['PassengerId', 'Survived']]
df.index = df['PassengerId']
df.drop('PassengerId', axis='columns', inplace=True)
df.sort_values(by=['Survived'], ascending=False, inplace=True)
print(df.head(15))
# df[0:total_alive_count] = int(1)
#
# df['Survived'] = df['Survived'].astype('int64')
print(df.info())
df.to_csv(path_or_buf='output.csv', header=True)