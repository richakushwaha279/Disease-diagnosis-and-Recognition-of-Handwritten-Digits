############################# ACTIVATION FUNCTIONS #####################################
def sigmoid(x, derive=False):
	old_settings = np.seterr(all='ignore')
	if derive:
		np.seterr(**old_settings)
		return x * (1 - x)
	else:
		f = 1 / (1 + np.exp(-x))
		np.seterr(**old_settings)
		return f

def ReLU(x):
    return np.where(x > 0, 1.0, 0.0)

def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis=1, keepdims=True)
