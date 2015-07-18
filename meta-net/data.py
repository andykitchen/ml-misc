import collections
import gzip, cPickle

import numpy

import theano
import theano.tensor as T

Data = collections.namedtuple('Data', ['x', 'y', 'size'])

def load_mnist(dataset):
	with gzip.open(dataset, 'rb') as f:
		train_raw, valid_raw, test_raw = cPickle.load(f)

	def theano_load(data_x, data_y):
		shared_x = theano.shared(numpy.asarray(data_x,
											   dtype=theano.config.floatX),
								 borrow=True)
		shared_y = theano.shared(numpy.asarray(data_y,
											   dtype=theano.config.floatX),
								 borrow=True)
		return Data(shared_x, T.cast(shared_y, 'int32'), data_x.shape[0])

	train = theano_load(*train_raw)
	valid = theano_load(*valid_raw)
	test  = theano_load(*test_raw)

	return train, valid, test

BatchedData = collections.namedtuple('Data', ['x', 'y', 'size', 'batch_size', 'batch_count'])

def batch(data, batch_size=100):
	return BatchedData(data.x, data.y, data.size, batch_size, batch_count=data.size/batch_size)

def build_validation_function(data, errors, x, y):
	batch_size  = data.batch_size
	batch_count = data.batch_count

	index = T.lscalar()

	errors_batch = theano.function(
		inputs=[index],
		outputs=errors,
		givens={
			x: data.x[index * batch_size: (index + 1) * batch_size],
			y: data.y[index * batch_size: (index + 1) * batch_size]
		}
	)

	def error_rate():
		total_errors = sum(map(errors_batch, range(batch_count)))
		return float(total_errors) / data.size

	return error_rate
