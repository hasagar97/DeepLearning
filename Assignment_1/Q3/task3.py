import tensorflow as tf
import numpy as np
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()


learning_rate = 0.2
training_iteration = 500
batch_size = 100
display_step = 2

train_x = train_x.reshape(train_x.shape[0], 28*28)
test_x = test_x.reshape(test_x.shape[0], 28*28)

n_dim = train_x.shape[1]
print("n_dim = ", n_dim)

n_class = 10


# cost_history = np.empty(shape=[1],dtype=float)


h=[60,60,60,60]

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

print("n_dim ",n_dim)


x = tf.placeholder(tf.float32, [None, n_dim],name="x_placeholder")
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y = tf.placeholder(tf.float32, [None, n_class])



def addLayer(inputx,inputdim,outputdim,activation):
	weights=tf.Variable(tf.truncated_normal([inputdim,outputdim]))
	biases=tf.Variable(tf.truncated_normal([outputdim]))

	layer=tf.add(tf.matmul(inputx,weights),biases)
	if activation:
		layer=activation(layer)
	print("Adding dense layer")
	return layer

# tf.Session().run(init)
y=addLayer(x,n_dim,h[0],tf.nn.relu)
y=addLayer(y,h[0],h[1],tf.nn.relu)
y=addLayer(y,h[1],h[2],tf.nn.relu)
y=addLayer(y,h[2],h[3],tf.nn.sigmoid)
model=addLayer(y,h[3],n_class,None)
y = tf.placeholder(tf.float32, [None, n_class])


init = tf.global_variables_initializer()
tf.Session().run(init)

with tf.name_scope("cost_function") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initializing the variables
init = tf.initialize_all_variables()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()


with tf.Session() as sess:
    sess.run(init)

    
    
    # Change this to a location on your computer
    summary_writer = tf.summary.FileWriter('data/logs', graph_def=sess.graph_def)

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # Write logs for each iteration
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        # Display logs per iteration step
        if iteration % display_step == 0:
            print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning completed!")

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print("Accuracy: ",accuracy.eval({x:mnist.text.images, y: mnist.test.labels}))
    print("life is strange")