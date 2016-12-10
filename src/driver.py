import numpy as np
import tensorflow as tf
import constant as c
from linearregression import linear_regression as lr
from feedforward import feed_forward as ff
import sys

TRAIN_FILE = "../data/train_prcsd.csv"
TEST_FILE = "../data/test_prcsd.csv"
TEST_LABEL_FILE = "../data/test_label_NN.txt"

#Neural Network Constants
NN_HIDDEN1 = 120
NN_HIDDEN2 = 60
NN_BATCH_SIZE = 500
NN_LEARINNG_RATE = 0.01

#Linear Regression Constants
LR_BATCH_SIZE = 1000
LR_LEARINNG_RATE = 0.01

#Select Model: True -> NN ; False -> LR
def driver_module(MODEL):
	print("train data loading...")
	train_data = np.genfromtxt(TRAIN_FILE, dtype=float, delimiter=',')
	print("train data loaded")
	print("training starts...")

	print("creating model...")
	if(MODEL):
		model = ff.create_model(NN_BATCH_SIZE, NN_HIDDEN1, NN_HIDDEN2, NN_LEARINNG_RATE)
		print("NN model with %d hidden1 size and %d hidden2 size created" %(NN_HIDDEN1, NN_HIDDEN2))
		train_and_test(ff, model, train_data, NN_BATCH_SIZE)
	else:
		model = lr.create_model(LR_BATCH_SIZE, LR_LEARINNG_RATE)
		print("LR model created")
		train_and_test(lr, model, train_data, LR_BATCH_SIZE)

#train the model against the loaded train data and
#test the model against the test data
def train_and_test(modeltype, model, train_data, batch_size):
    print("Initializing all model variables")
    model["session"].run(tf.initialize_all_variables())
    cur_batch = 0
    while (cur_batch < train_data.shape[0]):
        print("running batch: %d" % (cur_batch / batch_size))
        modeltype.run_training(model, train_data[cur_batch: cur_batch + batch_size, :])
        cur_batch += batch_size

	#freeing RAM for efficiency
    del train_data

    test_data = np.genfromtxt(TEST_FILE, dtype=float, delimiter=',')
    print("test data loaded")

    total_error_avg = 0
    cur_batch = 0
    while (cur_batch < test_data.shape[0]):
        y_pred = modeltype.run_test_wihthout_label(model, test_data[cur_batch: cur_batch + batch_size, :])
        y_tar = np.reshape(test_data[cur_batch: cur_batch + batch_size:, -1], (batch_size, c.NUM_OUTPUT))
        error_avg = np.mean(np.abs(y_pred - y_tar))
        total_error_avg += error_avg
        print("average error for batch:%d = %.3f" % ((cur_batch / batch_size), error_avg))
        cur_batch += batch_size
    print("average testing error = %.3f" % (total_error_avg / (test_data.shape[0] / batch_size)))

def main(argv):
    if(argv == "-lr"):
        driver_module(False)
    elif(argv == "-nn"):
        driver_module(True)
    else:
        print("usage driver.py -lr|-nn")
        sys.exit(2)

if __name__ == "__main__":
   main(sys.argv[1])