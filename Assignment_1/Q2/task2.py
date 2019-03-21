import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt


print("imported stuff")
def loadfiles():
	masses=np.load("masses.npy")
	positions=np.load("positions.npy")
	velocities=np.load("velocities.npy")
	return masses,positions,velocities
	# return 1,2,3

masses,positions,velocities=loadfiles()

massReshaped=masses.reshape(100,1,1)

x=tf.placeholder(dtype=tf.float64)
v=tf.placeholder(dtype=tf.float64)
m=tf.placeholder(dtype=tf.float64)


dt=tf.constant(0.0001,tf.float64)
G=tf.constant(6.67*100000,tf.float64)
x1=tf.expand_dims(x,0)
x2=tf.expand_dims(x,1)
print(x1.shape)



rmat=tf.subtract(x1,x2)#now r1's shape is (100,100,2),
rsqr=tf.pow(rmat,2)#sqaring(x2,y2)
rsum_=tf.reduce_sum(rsqr,axis=2,name="Square_sum")#adding sum x2+y2
rsum=tf.matrix_set_diag(rsum_,np.ones((100)))#making diagonal non zero
rsqrt=tf.pow(rsum,3/2)#sqrt(x2+y2)^3
rdistance=tf.reshape(rsqrt,[100,100,1])#distance matrix
r=tf.divide(rmat,rdistance)#kind of direction matrix

force=tf.multiply(m,r)#mulriplies by mass
acc=tf.reduce_sum(force,axis=0)#summation for particles
a=tf.multiply(acc,-G)#final accelertion

x_=tf.add(x,tf.multiply(v,dt))#updation for positon
xnew=tf.add(x_,tf.multiply(a,1/2*dt*dt))
vnew=tf.add(v,tf.multiply(a,dt))

thresh=tf.reduce_min(rsum)

CellSize=100

with tf.Session() as sess:
	# sess.run(tf.global_variables_initializer())
	a=sess.run(thresh,feed_dict={x: positions,v: velocities,m: massReshaped})
	i=0
	while(a>0.01):
		print("a: ",a," i: ",i)
		a,positions,velocities=sess.run([thresh,xnew,vnew],feed_dict={x: positions,v: velocities,m: massReshaped})
		# sess.run(tf.assign(x,xnew))
		# sess.run(tf.assign(v,vnew))
		i+=1
		# if (i%1==0): 
		# 	Cord=np.array(positions) # Tranfer particles cordinates to numpy array format to use in plotting functin
		# 	plt.clf()# clear figure   
		# 	plt.xlim(-CellSize, CellSize)# define figure axis size
		# 	plt.ylim(-CellSize, CellSize)# define figure axis size
		# 	plt.title(["step ",i])# figure  title
		# 	plt.scatter(Cord[:,0],Cord[:,1]);# add particle position to graph
		# 	plt.savefig("output/step_"+str(i))
	print("a: ",a," i: ",i )
	np.save('positionsFinal',positions)
	np.save('velocitiesFinal',velocities)
	writer=tf.summary.FileWriter("logss",sess.graph)