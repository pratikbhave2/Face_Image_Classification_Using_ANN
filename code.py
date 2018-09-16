
"""
Note : In order to run this code, you need to have the mat files "total_label.mat" and "total_data.mat".
These files can be created using "save_mat.py" file.
Order of running codes-
data_creation.py --> save_mat.py --> code.py
total_data contains 22,000 flattened images of dimension 60x60x3, size 22000x10800
total_label contains 22,0000 labels in one hot form ([0 1] for face and [1 0] for non face) of size 22000x2.
Description: This is the actual code for implementing the babysitting process of neural networks, taken from 
www.pythonprogramming.net, tweaked according to the requirements.
"""


import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


total_label = scipy.io.loadmat('total_label.mat')['total_label']
total_data= scipy.io.loadmat('total_data.mat')['total_data']

#divide the data into training and testing sets
train_data=total_data[1000:21000]
train_label=total_label[1000:21000]
test_data=np.concatenate((total_data[0:1000], total_data[21000:22000]),axis=0)
test_label=np.concatenate((total_label[0:1000], total_label[21000:22000]),axis=0)

#Neural Network Code

def epoch_data(start_num: int =100, batch_size: int =128):
    epoch_x=np.zeros([batch_size,10800])
    epoch_y=np.zeros([batch_size,2])
    for i in range(batch_size):
        epoch_x[i]=train_data[i+start_num,:]
        epoch_y[i]=train_label[i+start_num,:]
        
    return epoch_x, epoch_y

beta =0.001
n_nodes_hl1 = 2000
n_nodes_hl2 = 1000
#n_nodes_hl3 = 500
n_classes = 2
batch_size = 128

x = tf.placeholder('float', [None, 10800])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([10800, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

#    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
#                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.leaky_relu(l1)   
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.leaky_relu(l2)
    

    output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']
    regularization = tf.nn.l2_loss(hidden_1_layer['weights']) + \
                     tf.nn.l2_loss(hidden_2_layer['weights']) + \
                     tf.nn.l2_loss(output_layer['weights'])
    return output,regularization

def train_neural_network(x):
    prediction,regularization = neural_network_model(x)
    beta= 0.001
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    cost =  tf.reduce_mean(cost + beta*regularization)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    #optimizer = tf.train.AdadeltaOptimizer().minimize(cost)
    #optimizer = tf.train.AdagradOptimizer(0.001).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.01, use_locking= False).minimize(cost)
       
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./graphs',sess.graph)
        count = 0
        loss=[]
        for epoch in range(hm_epochs):
            epoch_loss = 0
            start_num=0
            for _ in range(int((len(train_data))/batch_size)):
                epoch_x, epoch_y=epoch_data(start_num,batch_size)
                #print(epoch_x, epoch_y)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                start_num +=100 
                count +=1
                loss.append(epoch_loss)
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_data, y:test_label}))
        writer.close()
    return count, loss
count, loss =train_neural_network(x)
loss=np.asarray(loss)
np.sort(loss)
loss[::-1].sort()
plt.plot(range(count),loss)
plt.ylabel("Loss")
plt.xlabel("Number of Epochs")
plt.title("Plot of Number of epochs vs Loss")
