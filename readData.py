import csv

# package name

# Custom Import


# read data from file to the dataset and training set
def read_data(fname, fname_tes):
	fd = open(fname,'r')
	freader = csv.reader(fd)
	global Dataset, Test_dataset
	Dataset = []
	Test_dataset = []

	if fname == 'dermatology.data':
		for line in freader:
			if '?' not in line  and len(line) != 0 and (line[-1] in ['1', '2', '3']):
				Dataset.append(map(float, line))
	if fname == 'pendigits.tra':
		for line in freader:
			if len(line) != 0 and (line[-1] in [' 0', ' 1', ' 2', ' 3']):
				Dataset.append(map(float, line))
		fd2 = open(fname_tes,'r')
		freader = csv.reader(fd2)
		for line in freader:
			if len(line) != 0 and (line[-1] in [' 0', ' 1', ' 2', ' 3']):
				Test_dataset.append(map(float, line))
