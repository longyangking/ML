import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

## TODO: Absorption Boundary Condition

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

a0 = 2.8e-4
b0 = 5e-3
tau0 = 0.1
k0 = -0.005
sess = tf.InteractiveSession()

N = 100
dx0 = 2.0/N 
T0 = 10.0
dt0 = 0.9*dx0**2/2
n = int(T0/dt0)

U_init = np.random.random((N,N)).astype('float32')
V_init =  np.random.random((N,N)).astype('float32')

#plt.figure()
#plt.imshow(field0)

dt = tf.placeholder(tf.float32,shape=())
k = tf.placeholder(tf.float32,shape=())
a = tf.placeholder(tf.float32,shape=())
b = tf.placeholder(tf.float32,shape=())
tau = tf.placeholder(tf.float32,shape=())
dx = tf.placeholder(tf.float32,shape=())

U = tf.Variable(U_init)
V = tf.Variable(V_init)

U_ = U + dt*(a*laplace(U)/dx/dx + U - U*U*U - V + k)
V_ = V + dt*(b*laplace(V)/dx/dx + U - V)/tau

step = tf.group(
    # Core
    U[1:-1,1:-1].assign(U_[1:-1,1:-1]),
    V[1:-1,1:-1].assign(V_[1:-1,1:-1]),
    # Boundary
    U[0,:].assign(U[1,:]),
    U[-1,:].assign(U[-2,:]),
    U[:,0].assign(U[:,1]),
    U[:,-1].assign(U[:,-2]),

    V[0,:].assign(V[1,:]),
    V[-1,:].assign(V[-2,:]),
    V[:,0].assign(V[:,1]),
    V[:,-1].assign(V[:,-2])
)

tf.initialize_all_variables().run()

print('Starting... with steps {n}'.format(n=n))

for i in range(n):
    step.run({a:a0,b:b0,tau:tau0,k:k0,dt:dt0,dx:dx0})

    if i%1000==0:
        print('Complete:{:.2f}%...'.format(i/n*100))

plt.imshow(U.eval(), cmap=plt.cm.copper, extent=[-1,1,-1,1])
plt.xticks([])
plt.yticks([])

plt.show()