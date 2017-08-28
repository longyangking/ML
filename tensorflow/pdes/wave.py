import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 


def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return tf.constant(a,dtype=1)

def simple_conv(x,k):
    x = tf.expand_dims(tf.expand_dims(x,0),-1)
    y = tf.nn.depthwise_conv2d(x,k,[1,1,1,1],padding='SAME')
    return y[0,:,:,0]

def laplace(x):
    laplace_k = make_kernel([[0.0,1.0,0.0],
                            [1.0,-4.0,1.0],
                            [0.0,1.0,0.0]])
    return simple_conv(x, laplace_k)

sess = tf.InteractiveSession()

N = 200

field0 = np.zeros((N,N),dtype='float32')
fieldp0 = np.zeros((N,N),dtype='float32')

field0[int(N/2),int(N/2)] = 1

plt.figure()
plt.imshow(field0)

dt = tf.placeholder(tf.float32,shape=())
c = tf.placeholder(tf.float32,shape=())

field = tf.Variable(field0)
fieldp = tf.Variable(field0)

field_ = dt*dt/c*laplace(field) + 2*field - fieldp
fieldp_ = field 

step = tf.group(
    field.assign(field_),
    fieldp.assign(fieldp_)
)

tf.initialize_all_variables().run()

for i in range(5000):
    step.run({dt:0.1,c:1.0})

    if i%100==0:
        print('Step:{step}...'.format(step=i))

plt.figure()
plt.imshow(field.eval()) 

plt.show()