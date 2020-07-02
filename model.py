import tensorflow as tf
import numpy as np

class MultiLayerPerceptrons():
    def __init__(self, x, num_inputs, num_outputs, num_layers, num_neurons):
        self.x = x
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_neurons = num_neurons

    def network(self):
        tf.reset_default_graph()
        w = []
        b = []
        for i in range(self.num_layers):
            # weights
            print ([self.num_inputs if i == 0 else self.num_neurons[i - 1],
                self.num_neurons[i]])
            w.append(tf.Variable(tf.random_normal(
                [self.num_inputs if i == 0 else self.num_neurons[i - 1],
                self.num_neurons[i]]),
                name="w_{0:04d}".format(i)
            ))
            # biases
            b.append(tf.Variable(tf.random_normal(
                [self.num_neurons[i]]),
                name="b_{0:04d}".format(i)
            ))
        w.append(tf.Variable(tf.random_normal(
            [self.num_neurons[self.num_layers - 1] if self.num_layers > 0 else self.num_inputs,
            self.num_outputs]), name="w_out"))
        b.append(tf.Variable(tf.random_normal([self.num_outputs]), name="b_out"))

        # x is input layer
        layer = self.x
        # add hidden layers
        for i in range(self.num_layers):
            layer = tf.nn.relu(tf.matmul(layer, w[i]) + b[i])
        # add output layer
        layer = tf.matmul(layer, w[self.num_layers]) + b[self.num_layers]

        return layer

    def fit(self, n_epochs, n_batches,
            batch_size, batch_func,
            model, optimizer, loss, accuracy_func,
            X_test, y_test):
        with tf.Session() as tfs:
            tfs.run(tf.global_variables_initializer())
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                for batch in range(n_batches):
                    X_batch, y_batch = batch_func(batch_size)
                    feed_dict = {x: X_batch, y: y_batch}
                    _, batch_loss= tfs.run([optimizer, loss], feed_dict)
                    epoch_loss += batch_loss
                average_loss= epoch_loss / n_batches
                print ("epoch: {0:04d} || loss: {1:0.6f}".format(
                    epoch, average_loss
                ))
            feed_dict = {x: X_test, y: y_test}
            accuracy_score = tfs.run(accuracy_func, feed_dict= feed_dict)
            print ("Accuracy: {0:.8f}".format(accuracy_score))
    

