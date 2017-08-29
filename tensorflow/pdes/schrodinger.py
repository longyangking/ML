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

a0 = 5.0e-3
V0 = 0
sess = tf.InteractiveSession()

N = 100
dx0 = 2.0/N 
T0 = 1.0
dt0 = 0.9*dx0**2/2
n = int(T0/dt0)

fr_init = np.zeros((N,N)).astype('float32')
fi_init = np.zeros((N,N)).astype('float32')
fr_init[int(N/2),int(N/2)] = 1/np.square(2)
fi_init[int(N/2),int(N/2)] = 1/np.square(2)

dt = tf.placeholder(tf.float32,shape=())
dx = tf.placeholder(tf.float32,shape=())
a = tf.placeholder(tf.float32,shape=())
V = tf.placeholder(tf.float32,shape=())

fr = tf.Variable(fr_init)
fi = tf.Variable(fi_init)

fr_ = fr + dt*(a*laplace(fi)/dx/dx + V*fi)
fi_ = fi - dt*(a*laplace(fr)/dx/dx + V*fr)

step = tf.group(
    # Core
    fr[1:-1,1:-1].assign(fr_[1:-1,1:-1]),
    fi[1:-1,1:-1].assign(fi_[1:-1,1:-1]),
    # BoTndary
    fr[0,:].assign(fr[1,:]),
    fr[-1,:].assign(fr[-2,:]),
    fr[:,0].assign(fr[:,1]),
    fr[:,-1].assign(fr[:,-2]),

    fi[0,:].assign(fi[1,:]),
    fi[-1,:].assign(fi[-2,:]),
    fi[:,0].assign(fi[:,1]),
    fi[:,-1].assign(fi[:,-2])
)

tf.initialize_all_variables().run()

print('Starting... with steps {n}'.format(n=n))

for i in range(n):
    step.run({a:a0,dt:dt0,dx:dx0,V:V0})

    if i%1000==0:
        print('Complete:{:.2f}%...'.format(i/n*100))

plt.imshow(fr.eval(),cmap='jet',interpolation="bicubic")
plt.xticks([])
plt.yticks([])

plt.show()