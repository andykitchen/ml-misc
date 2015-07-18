import numpy

import theano
import theano.tensor as T

def zero_matrix_bias(n_in, n_out):
	W = theano.shared(
		value=numpy.zeros(
			(n_in, n_out),
			dtype=theano.config.floatX
		),
		name='W',
		borrow=True
	)

	b = theano.shared(
		value=numpy.zeros(
			(n_out,),
			dtype=theano.config.floatX
		),
		name='b',
		borrow=True
	)

	return W, b

def tanh_matrix_bias(rng, n_in, n_out):
	W_values = numpy.asarray(
		rng.uniform(
			low=-numpy.sqrt(6. / (n_in + n_out)),
			high=numpy.sqrt(6. / (n_in + n_out)),
			size=(n_in, n_out)
		),
		dtype=theano.config.floatX
	)
	W = theano.shared(value=W_values, name='W', borrow=True)

	b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
	b = theano.shared(value=b_values, name='b', borrow=True)

	return W, b

def randn_matrix_bias(rng, n_in, n_out):
	W_values = numpy.asarray(
		0.05*rng.randn(n_in, n_out),
		dtype=theano.config.floatX
	)
	W = theano.shared(value=W_values, name='W', borrow=True)

	b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
	b = theano.shared(value=b_values, name='b', borrow=True)

	return W, b

def multinomial_logistic(x, W, b):
	p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)
	return p_y_given_x

def linear(x, W, b):
	return T.dot(x, W) + b

def nonlinear(activation):
	def linear(x, W, b):
		return activation(T.dot(x, W) + b)

	return linear

def compose(x, layers):
	params_list = []
	output = x
	for (layer_fn, params) in layers:
		output = layer_fn(output, *params)
		params_list.append(params)

	return output, params_list

def meta(x):
	rng = numpy.random.RandomState(1234)

	n_feats = 100
	feat_dim = 32
	feat_hidden = 128
	n_in = 28 * 28
	n_out = 10

	xm_values = numpy.asarray(
		numpy.sqrt(6. / (n_feats + feat_dim))*rng.randn(n_feats, feat_dim),
		dtype=theano.config.floatX
	)
	xm = theano.shared(value=xm_values, name='xm', borrow=True)

	meta_out, meta_params = compose(xm, [
		(nonlinear(T.tanh), tanh_matrix_bias(rng, feat_dim, feat_hidden)),
		(linear, tanh_matrix_bias(rng, feat_hidden, n_in)),
	])

	W1 = meta_out.T
	b1_values = numpy.zeros((n_feats,), dtype=theano.config.floatX)
	b1 = theano.shared(value=b1_values, name='b1', borrow=True)

	W2, b2 = tanh_matrix_bias(rng, n_feats, 10)

	output, params = compose(x, [
		(nonlinear(T.tanh), (W1, b1)),
		(nonlinear(T.nnet.softmax), (W2, b2)),
	])

	return output, [(xm,)] + meta_params + [(W2, b2)]

def mlp(x):
	rng = numpy.random.RandomState(1234)

	return compose(x, [
		(nonlinear(T.tanh), randn_matrix_bias(rng, n_in, n_hidden)),
		(nonlinear(T.nnet.softmax), randn_matrix_bias(rng, n_hidden, n_out)),
	])

def L2_penalty(W):
	return (W ** 2).sum()

def negative_log_likelihood(p_y_given_x, y):
	return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])

def errors(p_y_given_x, y):
	predictions = T.argmax(p_y_given_x, axis=1)
	return T.sum(T.neq(predictions, y))

def build_train_function(data, cost, x, y, params):
	index = T.lscalar()
	learning_rate = T.fscalar()
	grads = T.grad(cost=cost, wrt=params)
	updates = [(X, X - learning_rate * grad_X) for X, grad_X in zip(params, grads)]

	batch_size  = data.batch_size

	train_model = theano.function(
		inputs=[learning_rate, index],
		outputs=cost,
		updates=updates,
		givens={
			x: data.x[index * batch_size: (index + 1) * batch_size],
			y: data.y[index * batch_size: (index + 1) * batch_size]
		}
	)

	return train_model
