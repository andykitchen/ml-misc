import numpy as np
import collections

N = 1000
D = 50
Sigma = np.identity(D)
mu1 = np.zeros(D)
mu2 = np.zeros(D)

mu1[0:10]  = -1.0
mu1[10:20] = 1.0

x1 = np.random.multivariate_normal(mu1, Sigma, size=N/2)
x2 = np.random.multivariate_normal(mu2, Sigma, size=N/2)

x = np.concatenate([x1,x2], axis=0)
y = np.concatenate([np.repeat(0, N/2), np.repeat(1, N/2)], axis=0)

p = np.random.permutation(len(x))

x = x[p]
y = y[p]

DataSet = collections.namedtuple('DataSet', ['train', 'test'])
LabeledData = collections.namedtuple('LabeledData', ['x', 'y'])

n_test = 3*N/10
train_data = LabeledData(x=x[:-n_test], y=y[:-n_test])
test_data  = LabeledData(x=x[:n_test], y=y[:n_test])
data = DataSet(train=train_data, test=test_data)

LogisticModel = collections.namedtuple('LogisticModel', ['beta', 'alpha'])
alpha_init = np.array([.0])
# alpha_init = np.array([14.])
# beta_init = 0.01*np.random.randn(D)
beta_init = np.zeros(D)
# beta_init[0:10] = 1
# beta_init[10:20] = -1
model = LogisticModel(beta=beta_init, alpha=alpha_init)

def sigmoid(x):
	return 1./(1. + np.exp(-x))

def logistic_forward(m, x):
	return sigmoid(x.dot(m.beta) + m.alpha)

def test(m, d):
	p = logistic_forward(m, d.x)
	d = (p > .5) == d.y
	return d.mean()

def forward_loss(m, d):
	return loss(m, logistic_forward(m, d.x), d.y)

def loss(m, p, y):
	ent = -((1. - y)*np.log(1. - p) + y*np.log(p))
	return ent.mean()/np.log(2.)

def gradient_step(m, d, gamma, lam, tau):
	z = d.x.dot(m.beta) + m.alpha
	p = sigmoid(z)
	# dL_dy = (1. - d.y)/(1. - p) - d.y/p
	# dy_dz = p*(1. - p)
	# dL_dz = (1. - d.y)*p - d.y*(1 - p)
	dL_dz = p - d.y
	dL_dz = dL_dz[:, np.newaxis]
	dz_dalpha = 1.
	dz_dbeta = d.x
	dL_dalpha = (dL_dz * dz_dalpha).mean()
	dL_dbeta  = (dL_dz * dz_dbeta).mean(axis=0)
	np.subtract(m.alpha, gamma*dL_dalpha, out=m.alpha)
	# np.subtract(m.beta, gamma*dL_dbeta, out=m.beta)
	# np.subtract(m.beta, gamma*(dL_dbeta + lam*m.beta), out=m.beta)
	np.subtract(m.beta, gamma*(dL_dbeta + np.copysign(np.full_like(m.beta, lam), m.beta)), out=m.beta)
	if tau is not None:
		l = np.linalg.norm(m.beta)
		if l > tau:
			np.multiply(m.beta, tau/l, out=m.beta)
	return loss(m, p, d.y)

def run(epochs, learning_rate, regularization, norm_limit):
	epochs = int(epochs)
	loss = np.zeros(epochs)
	test_accuracy = np.zeros(epochs)

	for i in xrange(epochs):
		loss[i] = gradient_step(model, data.train, learning_rate, regularization, norm_limit)
		test_accuracy[i] = test(model, data.test)

	return (loss, test_accuracy)
