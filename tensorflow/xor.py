import tensorflow as tf 

learning_rate = 0.005

# Input layer
x_data = [[0.0,0.0], [1.0,0.0],[1.0,1.0],[0.0,1.0]]
x = tf.placeholder("float",shape=[None,2])

# Output layer
y_data = [0,1,0,1]
y = tf.placeholder("float",shape=[None,1])

weights = {
    'w1':tf.Variable(tf.random_normal([2,8])),
    'w2':tf.Variable(tf.random_normal([8,1]))
}

biases = {
    'b1':tf.Variable(tf.random_normal([1])),
    'b2':tf.Variable(tf.random_normal([1]))
}

def dnn(_X,_weights,_baises):
    d1 = tf.matmul(_X,_weights['w1'] + _baises['b1'])
    d1 = tf.nn.relu(d1)
    d2 = tf.matmul(d1,_weights['w2'] + _baises['b2'])
    d2 = tf.nn.sigmoid(d2)
    return d2

pred = dnn(x,weights,biases)

cost = tf.reduce_mean(tf.square(y - pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    for _ in range(3000):
        batch_xs = tf.reshape(x_data,shape=[-1,2])
        batch_ys = tf.reshape(y_data,shape=[-1,1])

        sess.run(optimizer,feed_dict={x:sess.run(batch_xs),
                y:sess.run(batch_ys)})
        acc = sess.run(accuracy,feed_dict={x:sess.run(batch_xs),
                y:sess.run(batch_ys)})
        loss = sess.run(cost,feed_dict={x:sess.run(batch_xs),
                y:sess.run(batch_ys)})
        step += 1
        if (step%100 == 0):
            print("Step "+str(step) + "   Loss {:.6f}".format(loss))
            print(sess.run(pred,feed_dict={x:sess.run(batch_xs)}))
        
    print("Optimization Finished!")