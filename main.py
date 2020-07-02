import tensorflow as tf
import pandas as pd
import numpy as np
from mnistLoader import mnistLoader
from model import MultiLayerPerceptrons
from sklearn.model_selection import train_test_split

tf.set_random_seed(123)
tf.reset_default_graph()

def mlp(x, num_inputs, num_outputs, num_layers, num_neurons):
    w = []
    b = []
    for i in range(num_layers):
        # weights
        print ([num_inputs if i == 0 else num_neurons[i - 1],
             num_neurons[i]])
        w.append(tf.Variable(tf.random_normal(
            [num_inputs if i == 0 else num_neurons[i - 1],
             num_neurons[i]]),
            name="w_{0:04d}".format(i)
        ))
        # biases
        b.append(tf.Variable(tf.random_normal(
            [num_neurons[i]]),
            name="b_{0:04d}".format(i)
        ))
    w.append(tf.Variable(tf.random_normal(
        [num_neurons[num_layers - 1] if num_layers > 0 else num_inputs,
         num_outputs]), name="w_out"))
    b.append(tf.Variable(tf.random_normal([num_outputs]), name="b_out"))

    # x is input layer
    layer = x
    # add hidden layers
    for i in range(num_layers):
        layer = tf.nn.relu(tf.matmul(layer, w[i]) + b[i])
    # add output layer
    layer = tf.matmul(layer, w[num_layers]) + b[num_layers]

    return layer

def tensorflow_classification(x, y, n_epochs, n_batches,
                              batch_size, batch_func,
                              model, optimizer, loss, accuracy_function,
                              X_test, Y_test):
    saver = tf.train.Saver()
    with tf.Session() as tfs:
        tfs.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch in range(n_batches):
                X_batch, Y_batch = batch_func(batch_size)
                feed_dict = {x: X_batch, y: Y_batch}
                _, batch_loss = tfs.run([optimizer, loss], feed_dict)
                epoch_loss += batch_loss
            average_loss = epoch_loss / n_batches
            print("epoch: {0:04d}   loss = {1:0.6f}".format(
                epoch, average_loss))
        feed_dict = {x: X_test, y: Y_test}
        accuracy_score = tfs.run(accuracy_function, feed_dict=feed_dict)
        print("accuracy={0:.8f}".format(accuracy_score))
        save_path = saver.save(tfs, "weights/tf-mnist.ckpt")

loader = mnistLoader()
X, y = loader.rawTrainLoader()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_test_ = loader.rawTestLoader()

def mnist_batch_func(batch_size=100):
    X_batch, Y_batch = loader.next_batch(batch_size, X_train, y_train)
    return [X_batch, Y_batch]

def save_csv(y):
    ImageId = list(np.asarray(list(range(28000))) + 1)
    Label = list(y)
    dict_value = {
        "ImageId" : ImageId,
        "Label" : Label
    }
    pf = pd.DataFrame(dict_value, columns=["ImageId", "Label"])
    pf.to_csv(" tf-mnist.csv", index= None)

def main():
    num_inputs = 784
    num_outputs = 10

    tf.reset_default_graph()

    num_layers = 2
    num_neurons = []
    for i in range(num_layers):
        num_neurons.append(256)
    
    learning_rate = 0.01
    n_epochs = 40
    batch_size = 128
    n_batches = int(len(X_train) / batch_size)

   # input images
    x = tf.placeholder(dtype=tf.float32, name="x", shape=[None, num_inputs])
    # target output
    y = tf.placeholder(dtype=tf.float32, name="y", shape=[None, num_outputs])

    model = mlp(x=x,
                num_inputs=num_inputs,
                num_outputs=num_outputs,
                num_layers=num_layers,
                num_neurons=num_neurons)

    # loss function
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    # optimizer function
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss)

    predictions_check = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy_function = tf.reduce_mean(tf.cast(predictions_check, tf.float32))

    # tensorflow_classification(x, y, n_epochs=n_epochs,
    #                         n_batches=n_batches,
    #                         batch_size=batch_size,
    #                         batch_func=mnist_batch_func,
    #                         model=model,
    #                         optimizer=optimizer,
    #                         loss=loss,
    #                         accuracy_function=accuracy_function,
    #                         X_test=X_test,
    #                         Y_test=y_test
    #                         )
    saver = tf.train.Saver()
    with tf.Session() as tfs:
        saver.restore(tfs, "weights/tf-mnist.ckpt")
        feed_dict = {x: X_test_}
        classification = tfs.run(tf.argmax(model, 1), feed_dict)
        save_csv(list(classification))

if __name__ == "__main__":
    main()