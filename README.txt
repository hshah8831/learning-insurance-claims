Dependency
	Python 3.4.2
	Tensorflow 0.10.0rc0
	Numpy

Run Information
	cd src                 //IMPORTANT: the code fails if it is not run from inside src directory  
	python3 driver.py -lr  //to run linear regression model
	python3 driver.py -nn  //to run neural network model

Model Parameters
	model parameters like number of neurons in layer 1 and 2(NN_HIDDEN1, NN_HIDDEN2), learning rate batch size (NN_LEARNING_RATE, LR_LEARNING_RATE) etc. 
	can be changed in the driver.py program 
	
Log Information
	The code prints out the batch wise cost at (step % (max_step/2)) == 0, where 
		step is epoch 
		max_step is maximum number of epochs allowed per batch
	
	The code also prints the testing information after the complete training cycle is over, and gives the average test error.
	
Current limitation:
	Cannot provide input file name, batch size from the command line. They are hard coded in the code.
	Batch size should be exact multiple of the total number of records. (The placeholder in tensorflow has to be 
	given exact batch size before the program runs, and cannot be changed in the middle of a session)