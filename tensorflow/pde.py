import tensorflow as tf 
import numpy as np 

#import PIL.Image
#from io import StringIO
#from IPython.display import clear_output, Image, display

import matplotlib.pyplot as plt

#def DisplayArray(a,fmt='jpeg',rng=[0,1]):
#    a = (a - rng[0])/float(rng[1] - rng[0])*255
#    a = np.uint8(np.clip(a,0,255))
#    f = StringIO()
#    PIL.Image.fromarray(a).save(f, fmt)
#    display(Image(data=f.getvalue()))

#def DisplayArrayPLT(a):
#    plt.imshow(a)
#    plt.show()

sess = tf.InteractiveSession()

def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1,1])
    return tf.constant(a,dtype=1)

def simple_conv(x,k):
    x = tf.expand_dims(tf.expand_dims(x,0),-1)
    y = tf.nn.depthwise_conv2d(x,k,[1,1,1,1],padding='SAME')
    return y[0,:,:,0]

def laplace(x):
    laplace_k = make_kernel([[0.5,1.0,0.5],
                            [1.0,-6.0,1.0],
                            [0.5,1.0, 0.5]])
    return simple_conv(x,laplace_k)

N = 250
u_init = np.zeros([N,N],dtype='float32')
ut_init = np.zeros([N,N],dtype='float32')

for n in range(40):
    a,b = np.random.randint(N,size=2)
    u_init[a,b] = np.random.uniform()

plt.figure()
plt.imshow(u_init)
plt.show(False)

eps = tf.placeholder(tf.float32,shape=())
damping = tf.placeholder(tf.float32,shape=())

U = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

U_ = U + eps*Ut
Ut_ = Ut + eps*(laplace(U) - damping*Ut)

step = tf.group(
    U.assign(U_),
    Ut.assign(Ut_)
)

tf.initialize_all_variables().run()


plt.figure()
for i in range(500):
    step.run({eps:0.03,damping:0.04})
    if i%50 == 0:
        print("Step:{step}...".format(step=i))
    
plt.imshow(U.eval())
plt.show()

