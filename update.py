import 

# package name

# Custom Import

# create folds from the training set
def create_folds():
	n = len(Dataset)/5
	for l in range(5):
		temp_fold = []
		j = 0
		while j < n:
			i = random.randrange(len(Dataset))
			temp_fold.append(Dataset.pop(i))
			j = j+1
		folds.append(temp_fold)
	#now Dataset is empty and folds are created

# update the Dataset with new training set of features after every fold
def updateDataset():
	global Dataset
	Dataset = []
	for x in folds:
		for y in x:
			Dataset.append(y)

# update the actual labels and their respective one hot vectors
def updateY(flag, fname):
	Y_labels = []
	if flag == True:
		for i in Dataset:
			Y_labels.append(i[-1])
	else:
		for i in Test_dataset:
			Y_labels.append(i[-1])
	
	# make Y_labels one hot vectors
	Y = []
	for i in Y_labels:
		if fname == 'dermatology.data':
			if i == 1:
				Y.append([1,0,0])
			elif i == 2:
				Y.append([0,1,0])
			elif i == 3:
				Y.append([0,0,1])
			else:
				print "wrong label"
		else:
			if i == 0:
				Y.append([1,0,0,0])
			elif i == 1:
				Y.append([0,1,0,0])
			elif i == 2:
				Y.append([0,0,1,0])
			elif i == 3:
				Y.append([0,0,0,1])
			else:
				print "wrong label"

	return Y, Y_labels

# update the feature vector matrix
def updateX(flag):
	X = []
	global Dataset
	if flag == True:
		for i in Dataset:
			l = []
			for j in i:
				l.append(j)
			l.pop(-1)
			X.append(l)
	else:
		for i in Test_dataset:
			l = []
			for j in i:
				l.append(j)
			l.pop(-1)
			X.append(l)
	return X
