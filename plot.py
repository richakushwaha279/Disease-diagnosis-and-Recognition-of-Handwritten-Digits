# plot the accuracy v/s no of hidden nodes
def plot_accuracy():
	mp.plot(hidden_nodes, accuracy_sigmoid, color = 'green', label = 'accuracy of sigmoid')
	mp.plot(hidden_nodes, accuracy_ReLU, color = 'blue', label = 'accuracy of ReLU')
	mp.xlabel("No. of hidden nodes", fontsize = 20)
	mp.ylabel("Accuracy", fontsize = 20)
	mp.legend(loc = 'upper right')
	mp.show()

# plot the loss of sigmoid and ReLU for 10 hidden nodes for dermatology dataset and 170 hidden nodes for pendigits dataset
def plot_costs(fname):
	if fname == 'dermatology.data':
		mp.plot(costs_sigmoid, color = 'green', label = 'loss using sigmoid with 10 hidden nodes & 5 fold-cross-validation')
		mp.plot(costs_ReLU, color = 'blue', label = 'loss using ReLU with 170 hidden nodes & 5 fold-cross-validation')
		mp.ylabel("Cross Entropy Loss", fontsize = 20)
		mp.legend(loc = 'upper right')
		mp.show()
	else:
		mp.plot(costs_sigmoid, color = 'green', label = 'loss using sigmoid with 10 hidden nodes')
		mp.plot(costs_ReLU, color = 'blue', label = 'loss using ReLU with 170 hidden nodes')
		mp.ylabel("Cross Entropy Loss", fontsize = 20)
		mp.legend(loc = 'upper right')
		mp.show()
