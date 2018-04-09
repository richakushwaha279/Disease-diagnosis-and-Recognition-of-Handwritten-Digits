def main():
	global accuracy_sigmoid, accuracy_ReLU, costs_sigmoid, costs_ReLU
	############################# read data #############################
	fname = raw_input('Enter Dataset file name: ')
	tesfname = ''
	if fname == 'pendigits.tra':
		tesfname = raw_input('Enter Testing Dataset file name: ')
	print
	########################## apply neural network #####################
	for a in activation_functions:
		if a == 'sigmoid':
			print "For sigmoid activation function: "
			accuracy_sigmoid = []
			costs_sigmoid = []
		else:
			print "For ReLU activation function: "
			accuracy_ReLU = []
			costs_ReLU = []
		print 'No. of hidden nodes v/s accuracy:'
		for d in hidden_nodes:
			read_data(fname, tesfname)
			print "\nNo. of hidden nodes: ", d,
			neural_network(fname, d, a)
		print

	plot_accuracy()
	plot_costs(fname)

if __name__ == '__main__':
	main()
