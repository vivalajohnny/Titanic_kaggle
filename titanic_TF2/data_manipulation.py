import pandas as pd

pd.set_option('display.width', 256)


'''Чтение инпут файлов. Файл train содержит данные о пассажирах с колонкой кто из них выжил.
Файл test содержит данные о других пассажирах, но без колонки выживаемости.'''

train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')

'''Объединим два файла в один большой. Полная выборка данных будет давать более
точные данные на этапе анализа данных (более точные медианные значения, средние, количественные
итд)'''
# all_data = pd.concat([train_data, test_data], sort=False)
# print(all_data.info())
'''Анализ данных можно проводить с помощью ручной подготовки данных и вывода их
через matplotlib, использовать все готовое в seaborn и текстовый вывод pandas.
    Выведем статистику выживаемости по полу и классу.'''

# print('============ Survived by Pclass and Sex: ')
# print(train_data.groupby(['Pclass', 'Sex'])['Survived'].value_counts(normalize=True)) # Normalize позволяет перевести количество выживших в проценты.

def for_cabin1(value):
    return str(value)

def for_cabin(value):
    try:
        int(value)
        return int(value)
    except ValueError:
        return int(0)

def embarked_transformation(letter):
    if letter == 'S':
        return int(1)
    if letter == 'C':
        return int(2)
    if letter == 'Q':
        return int(3)

def floor_transformation(letter):
    if letter == 'A':
        return 1
    if letter == 'B':
        return 2
    if letter == 'C':
        return 3
    if letter == 'D':
        return 4
    if letter == 'E':
        return 5
    if letter == 'F':
        return 6
    if letter == 'G':
        return 7
    return 0 if letter == 0 else 8

def get_title(x):
    '''Функция получения статуса пассажира из имени'''
    return x.split(', ')[1].split('.')[0]

def data_manipulation(data):
    '''Функция обработки столбцов (приведение к числовому виду,
    заполнение значений NaN медианными значениями, создание столбцов с новыми признаками итд)'''
    # PassengerId    1309 non-null int64
    # Survived       891 non-null float64
    # Pclass         1309 non-null int64
    # Name           1309 non-null object + - Достать статус человека (mr ms итд), добавить колонку со статусами +
    # Sex            1309 non-null object + - Приведение к числовому виду (мужчина - 1, женщина - 0) +
    # Age            1046 non-null float64 + - Разбить по бакетам +
    # (0 - младенец, 1 - ребенок, 2 - Подросток, 3 - Молодой, 4 - Средний, 5 - Взрослый, 6 - Старик.)
    # SibSp          1309 non-null int64 + - Создать колонку общее количество близких +
    # Parch          1309 non-null int64 + - Создать колонку общее количество близких +
    # Ticket         1309 non-null object +
    # Fare           1308 non-null float64 + Незаполненные значения заполнить медианными по классу +
    # Cabin          295 non-null object + Вычленить букву кабины (которая значит этаж на титанике), заполнение нан - 0+
    # Embarked       1307 non-null object + Привести к числовому виду +
    # Title          891 non-null object + Привести к числовому виду (нужно ли?) -
    # Проверить выживаемость по номеру кабины +
    # Добавить колонку с родителем (0 - нет, 1 - да) +
    # Добавить колонку с ребенком (0 - нет, 1 - да) +
    # Добавить колонку с количеством родственников +
    # Добавить столбец одиночка ли человек +
    # Добавить столбец с этажем на титанике, неизвестные этажи заполнить в соответствии с классом +

    # Обработка столбца Name (Создание нового столбца со статусами)
    data['Title'] = data['Name'].apply(get_title)
    # Обработка столбца Sex
    data.loc[data['Sex'] == 'male', 'Sex'] = 1
    data.loc[data['Sex'] == 'female', 'Sex'] = 0
    # Обработка столбца Age (Возраст меньше 1 - приравнять к 1. Больше 60 - к 60. Далее разделить все на бакеты.)
    data.loc[data['Age'] < 1, 'Age'] = 1
    data['Age'] = data['Age'] // 5
    data.loc[data['Age'] == 4, 'Age'] = 3
    data.loc[(data['Age'] > 4) & (data['Age'] < 9), 'Age'] = 4
    data.loc[(data['Age'] > 8) & (data['Age'] < 12), 'Age'] = 5
    data.loc[data['Age'] > 11, 'Age'] = 6
    data['Age'] = data['Age'].apply(for_cabin)
    # Cоздание столбца детей с родителями
    data['childs_w_parents'] = [0 for i in range(len(data))]
    data.loc[(data['Age'] < 3) & (data['Parch'] > 0), 'childs_w_parents'] = 1
    # Cоздание столбца родителя с детьми
    data['parents_w_childs'] = [0 for i in range(len(data))]
    data.loc[(data['Age'] > 2) & (data['Parch'] > 2), 'parents_w_childs'] = 1
    data.loc[(data['Age'] > 3) & (data['Parch'] > 0), 'parents_w_childs'] = 1
    # Создание колонки с общим количеством родственников
    data['relatives'] = data[['Parch', 'SibSp']].sum(1)
    # Создание колонки человек-одиночка
    data['single'] = data['relatives'].apply(lambda r: 1 if r == 0 else 0)
    # Создание столбца с этажем, где находился пассажир, плюс присваивание неизвестных к медианным
    data['Cabin'] = data['Cabin'].fillna(0)
    data['floor'] = data['Cabin'].apply(lambda r: r[0] if r != 0 else 0)
    data['floor'] = data['floor'].apply(floor_transformation)
    data.loc[(data['Pclass'] == 1) & (data['floor'] == 0), 'floor'] = 1
    data.loc[(data['Pclass'] == 2) & (data['floor'] == 0), 'floor'] = 5
    data.loc[(data['Pclass'] == 3) & (data['floor'] == 0), 'floor'] = 6
    # Приведение колонки эмбаркед к числовому виду
    data['Embarked'] = data['Embarked'].apply(embarked_transformation)
    data['Embarked'] = data['Embarked'].apply(for_cabin)
    # Приведение колонки Fare к числовому виду (деление по бакетам по квантилям)
    q25 = float(data['Fare'].quantile(q=0.25))
    q50 = float(data['Fare'].quantile(q=0.50))
    q75 = float(data['Fare'].quantile(q=0.75))
    q100 = float(data['Fare'].quantile(q=1))
    data.loc[data['Fare'] <= q25, 'Fare'] = 100000
    data.loc[data['Fare'] <= q50, 'Fare'] = 200000
    data.loc[data['Fare'] <= q75, 'Fare'] = 300000
    data.loc[data['Fare'] <= q100, 'Fare'] = 400000
    data['Fare'] = data['Fare'] / 100000
    data['Fare'] = data['Fare'].apply(for_cabin)
    # Добавление новой колонки с номером кабины
    data['cabin_number'] = data['Cabin'].fillna(str(0))
    data['cabin_number'] = data['cabin_number'].apply(for_cabin1)
    data['cabin_number'] = data['cabin_number'].apply(lambda r: r.split(' ')[0] if r != 0 else r)
    data['cabin_number'] = data['cabin_number'].apply(lambda r: r[1:] if r != 0 else r)
    data['cabin_number'] = data['cabin_number'].apply(for_cabin)
    data.loc[data['cabin_number'] == 0, 'cabin_number'] = 150
    data['cabin_number'] = data['cabin_number'] // 10
    data.loc[data['cabin_number'] == 15, 'cabin_number'] = -1
    return data


