import

# package name

# Custom Import


# feed forward and backpropagation and testing 
def neural_network(fname, d, a):
	global Test_dataset, Dataset, folds, accuracy_sigmoid, accuracy_ReLU
	if fname == 'dermatology.data':
		create_folds()
		costs = []
		accuracy = 0
		for i in folds:
			l = []
			Test_dataset = i
			folds.remove(i)
			c = 3 # no of classes
			w1 = np.random.randn(len(folds[0][0])-1, d)
			b1 = np.random.randn(d)
			w2 = np.random.randn(d, c)
			b2 = np.random.randn(c)
			alpha = 0.0005
			updateDataset()
			# True to update using Dataset
			X = updateX(True)
			Y, Y_labels = updateY(True, fname)
			
			for epoch in range(maxEpochs):
				# forward pass
				H = np.dot(X, w1) + b1
				if a == 'sigmoid':
					Act_H = sigmoid(H)
				else:
					Act_H = ReLU(H)
				O = np.dot(Act_H, w2) + b2
				Act_O = softmax(O)

				# backward pass
				delta2 = Act_O - Y
				w2 = w2 - alpha * Act_H.T.dot(delta2)
				b2 = b2 - alpha * (delta2).sum(axis=0)

				delta1 = (delta2).dot(w2.T) * Act_H * (1 - Act_H)
				w1 = w1 - alpha * np.asarray(X).T.dot(delta1)
				b1 = b1 - alpha * (delta1).sum(axis=0)
				if epoch % 100 == 0:
					loss = (np.sum(-np.asarray(Y) * np.log(np.asarray(Act_O))))/len(Dataset)
					if loss <= 0.02:
						print "\n***Cross Entropy Error crossed minimum threshold at epoch: ", epoch, "***"
						break
					if d == 10 and a == 'sigmoid':
						costs_sigmoid.append(loss)
					if d == 10 and a == 'ReLU':
						costs_ReLU.append(loss)
					# print loss
			
			# False to update using Test_dataset 
			X = updateX(False)
			Y, Y_labels = updateY(False, fname)		
			
			H = np.dot(X, w1) + b1
			if a == 'sigmoid':
				Act_H = sigmoid(H)
			else:
				Act_H = ReLU(H)
			O = np.dot(Act_H, w2) + b2
			Act_O = softmax(O)
			Act_O_labels = []
			Act_O_labels = np.argmax(Act_O, axis=1)
			
			curr_accuracy = 0

			for i in range(len(Y_labels)):
				if Y_labels[i] == Act_O_labels[i]+1:
					curr_accuracy = curr_accuracy + 1

			accuracy = accuracy + float(curr_accuracy)/len(Act_O_labels)
			folds.append(Test_dataset)

		accuracy = float(accuracy)/5
		accuracy = accuracy*100
		if a == 'sigmoid':
			accuracy_sigmoid.append(accuracy)
		else:
			accuracy_ReLU.append(accuracy)

		print "Accuracy: %.3f" %(accuracy)
	else:
		c = 4  # no of classes
		w1 = np.random.randn(len(Dataset[0])-1, d)
		b1 = np.random.randn(d)
		w2 = np.random.randn(d, c)
		b2 = np.random.randn(c)

		# True to update using Dataset
		X = updateX(True)
		Y, Y_labels = updateY(True, fname)
		
		for epoch in range(maxEpochs):
			# forward pass
			H = np.dot(X, w1) + b1
			if a == 'sigmoid':
				Act_H = sigmoid(H)
			else:
				Act_H = ReLU(H)
			O = np.dot(Act_H, w2) + b2
			Act_O = softmax(O)

			# backward pass
			alpha = 0.0005
			delta2 = Act_O - Y
			w2 = w2 - alpha * Act_H.T.dot(delta2)
			b2 = b2 - alpha * (delta2).sum(axis=0)

			delta1 = (delta2).dot(w2.T) * Act_H * (1 - Act_H)
			w1 = w1 - alpha * np.asarray(X).T.dot(delta1)
			b1 = b1 - alpha * (delta1).sum(axis=0)

			if epoch % 100 == 0:
				loss = (np.sum(-np.asarray(Y) * np.log(np.asarray(Act_O))))/len(Dataset)
				if loss <= 0.09:
					print "\n***Cross Entropy Error crossed minimum threshold at epoch: ", epoch, "***"
					break
				if d == 170 and a == 'sigmoid':
					costs_sigmoid.append(loss)
				if d == 170 and a == 'ReLU':
					costs_ReLU.append(loss)
				# costs.append(loss)
				# print loss

		# False to update using Test_dataset 
		X = updateX(False)
		Y, Y_labels = updateY(False, fname)	
		H = np.dot(X, w1) + b1
		if a == 'sigmoid':
			Act_H = sigmoid(H)
		else:
			Act_H = ReLU(H)
		O = np.dot(Act_H, w2) + b2
		Act_O = softmax(O)
		Act_O_labels = []
		Act_O_labels = np.argmax(Act_O, axis=1)
		
		accuracy = 0

		for i in range(len(Y_labels)):
			if Y_labels[i] == Act_O_labels[i]:
				accuracy = accuracy + 1

		accuracy = float(accuracy)/len(Act_O_labels)
		accuracy = accuracy*100

		if a == 'sigmoid':
			accuracy_sigmoid.append(accuracy)
		else:
			accuracy_ReLU.append(accuracy)

		print "Accuracy: %.3f" %accuracy

	Dataset = []
	Test_dataset = []
	folds = []
