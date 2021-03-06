import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

## TODO: Absorption BoTndary Condition

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

a0 = 1.0e-2
sess = tf.InteractiveSession()

N = 100
dx0 = 2.0/N 
T0 = 5.0
dt0 = 0.9*dx0**2/2
n = int(T0/dt0)

T_init = np.zeros((N,N)).astype('float32')
T_init[int(N/2),int(N/2)] = 1   # High TemperatTre

dt = tf.placeholder(tf.float32,shape=())
dx = tf.placeholder(tf.float32,shape=())
a = tf.placeholder(tf.float32,shape=())
T = tf.Variable(T_init)

T_ = T + dt*(a*laplace(T)/dx/dx)

step = tf.group(
    # Core
    T[1:-1,1:-1].assign(T_[1:-1,1:-1]),
    # BoTndary
    T[0,:].assign(T[1,:]),
    T[-1,:].assign(T[-2,:]),
    T[:,0].assign(T[:,1]),
    T[:,-1].assign(T[:,-2])
)

tf.initialize_all_variables().run()

print('Starting... with steps {n}'.format(n=n))

for i in range(n):
    step.run({a:a0,dt:dt0,dx:dx0})

    if i%1000==0:
        print('Complete:{:.2f}%...'.format(i/n*100))

plt.imshow(T.eval(),cmap="hot", extent=[-1,1,-1,1])
#plt.colorbar()
plt.xticks([])
plt.yticks([])

plt.show()