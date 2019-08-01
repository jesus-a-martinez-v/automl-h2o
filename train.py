import h2o
from h2o.automl import H2OAutoML

print('[INFO] Initializing H2O.')
h2o.init()

print('[INFO] Loading MNIST dataset.')
train = h2o.import_file('./data/mnist_train.csv')
test = h2o.import_file('./data/mnist_test.csv')

x = train.columns
y = 'label'
x.remove(y)

print('[INFO] Running automatic machine learning process...')
automl = H2OAutoML(max_models=30, seed=42)
automl.train(x=x, y=y, training_frame=train)

print('[INFO] Models leader board:')
leader_board = automl.leaderboard
print(leader_board.head(rows=leader_board.nrow))

print('[INFO] Leader/best model:')
print(automl.leader)

predictions = automl.predict(test)
