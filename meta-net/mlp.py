import numpy

import theano
import theano.tensor as T

import data
import model

import itertools

batch_size = 10

train_data, valid_data, test_data = data.load_mnist('mnist.pkl.gz')

index = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')

p_y_given_x, layer_params = model.meta(x)
params = list(itertools.chain(*layer_params))
cost = model.negative_log_likelihood(p_y_given_x, y)

errors = model.errors(p_y_given_x, y)
validate_model = data.build_validation_function(data.batch(valid_data, batch_size=1000), errors, x, y)

n_epochs = 500
learning_rate = 0.01
L1_lambda = 0.001
L2_lambda = 0.001

train_batched = data.batch(train_data, batch_size)
train_model = model.build_train_function(train_batched, 
	cost + L1_lambda*T.sum(abs(params[0])) + L2_lambda*T.sum(params[1]**2), x, y, params)

def save_model(name):
	import cPickle
	with open(name+'-params.pkl', 'wb') as f:
	    cPickle.dump(params, f)

for epoch in range(n_epochs):
	if epoch > 0 and (epoch % 100) == 0:
		learning_rate *= 0.9
		print "-- decreased learning rate: ", learning_rate

	print validate_model()
	for i in range(train_batched.batch_count):
		train_model(learning_rate, i)
