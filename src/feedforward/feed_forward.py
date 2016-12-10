import tensorflow as tf
import numpy as np
import constant as c
from feedforward import neural_net as nn
import constant as c

#gives tensorflow placeholder with given batch size
def placeholder_inputs(batch_size):
  features_placeholder = tf.placeholder(tf.float32, shape=(batch_size, c.NUM_FEATURES))
  labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, c.NUM_OUTPUT))
  return features_placeholder, labels_placeholder

#returns a feed_dict with only features, which is used sess.run()   
def fill_feed_dict_without_labels(data_set, batch_size, features_pl):
  feature = np.reshape(data_set[:,0:c.NUM_FEATURES],(batch_size, c.NUM_FEATURES))

  feed_dict = {
      features_pl: feature,
  }
  return feed_dict

#returns a feed_dict with features and labels, which is used sess.run()
def fill_feed_dict(data_set, batch_size, features_pl, labels_pl):
    features = np.reshape(data_set[:,0:c.NUM_FEATURES],(batch_size, c.NUM_FEATURES))
    label = np.reshape(data_set[:, -1], (batch_size, c.NUM_OUTPUT))
    feed_dict = {
      features_pl: features,
      labels_pl: label,
    }
    return feed_dict

#returns a model which has grouped session, all the operations (training, cost) and 
#the placeholder for the calling function to use
def create_model(batch_size, hidden1, hidden2, learning_rate):
    tf.Graph().as_default()
    # Generate placeholders for the features and labels.
    features_placeholder, labels_placeholder = placeholder_inputs(batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = nn.inference(features_placeholder, hidden1, hidden2)

    # Add to the Graph the Ops for loss calculation.
    cost_op = nn.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = nn.training(cost_op, learning_rate)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = None
    #summary = tf.merge_all_summaries()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()
    model = {"session" : sess, "placeholders" : [features_placeholder, labels_placeholder], "operations" : [logits, cost_op, train_op], "summary":summary}
    return model

#use the incoming model and data set (batch size * number of features) to train
#max_stp number of times
def run_training(model, data_set):
    features_placeholder, labels_placeholder = model["placeholders"]
    cost_op = model["operations"][1]
    train_op = model["operations"][2]
    sess = model["session"]
    # Start the training loop.
    step = 0
    loss_value = 100
    while(loss_value > c.TOLERENCE and step < c.MAX_STEPS):
        feed_dict = fill_feed_dict(data_set, data_set.shape[0], features_placeholder, labels_placeholder)
		#runs the training operation and cost operation sequentially
        _, loss_value = sess.run([train_op, cost_op], feed_dict=feed_dict)

        if step % (c.MAX_STEPS/2) == 0:
            #Print status to stdout.
            print('Step %d: loss = %.2f' % (step, loss_value ))

        step += 1

#returns the predicted value for the given data set
#uses fill_feed_dict_without_labels function to use only the 
#features from the given data set
def run_test_wihthout_label(model, data_set):
    features_placeholder, labels_placeholder = model["placeholders"]
    logits = model["operations"][0]
    sess = model["session"]
    feed_dict = fill_feed_dict_without_labels(data_set, data_set.shape[0], features_placeholder)
	#runs the logits operations to get the predicted value from forward propagation of the neural network 
    y_pred = sess.run(logits, feed_dict=feed_dict)
    return y_pred
