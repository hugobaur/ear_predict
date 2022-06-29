import pandas as pd
import numpy as np
import logging
#
# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras.wrappers.scikit_learn import KerasRegressor


df = pd.read_csv('rdh_database.csv')
df_tucurui = df[(df.id_posto == 275)]
df_tucurui['id_tempo']= pd.to_datetime(df_tucurui['id_tempo'],format='%Y%m%d').dt.date


def get_train_test(df, split_percent=0.50):
    # df = read_csv(url, usecols=[1], engine='python')
    data = np.array(df.values.astype('float32'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = np.array(data).reshape(-1,1)

    data = scaler.fit_transform(data).flatten()
    n = len(data)
    # Point for splitting data into train and test
    split = int(n*split_percent)
    train_data = data[range(split)]
    test_data = data[split:]
    return train_data, test_data, data


train_data_vazao, test_data_vazao, data_vazao = get_train_test(df_tucurui['vazao_natural'])


def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape,
                        activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
    return model

# model = create_RNN(10, 3, (1,1), activation=['tanh', 'tanh'])


scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
        }


params={'hidden_units':[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        'dense_units':[1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        'input_shape':[(3,1),(4,1),(5,1),(6,1),(7,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)],
        'activation':[('relu','relu'),('tanh','tanh'),('linear','linear')]}
# 'sigmoid',
# param_grid = dict(hidden_units=hidden_units, time_steps=time_steps,dense_units=dense_units,input_shape=input_shape,activation=activation)

model = KerasRegressor(build_fn=create_RNN, verbose=0)

gs=GridSearchCV(estimator=model, param_grid=params, cv=10, verbose=0, scoring=scorers,refit="precision_score", n_jobs=-1)
# now fit the dataset to the GridSearchCV object.
# gs = gs.fit(train_data_vazao, test_data_vazao).score(train_data_vazao, test_data_vazao)
gs = gs.fit(train_data_vazao, test_data_vazao, verbose = 0)

print('Melhores params: ' + str(gs.best_params_))
print('Accuracy: ' + str(gs.best_score_))


# if __name__ == '__main__':
#     print(df)


