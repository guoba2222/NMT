# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:13:54 2018

@author: Administrator
"""
###### import ##############
import time
from collections import namedtuple
import numpy as np
import tensorflow as tf

###### Data processing——load data,encode ##############
with open('anna.txt', 'r') as f:
    text = f.read()
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], np.int32)

###### get_batches ##############
def get_batches(arr, n_seqs, n_steps):
    BatchSize = n_seqs*n_steps
    n_BatchSize = int(len(arr)/BatchSize)
    arr = arr[:BatchSize*n_BatchSize]
    arr = arr.reshape((n_seqs,-1))

    for n in range(0,arr.shape[1],n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)

        y[:, :-1], y[:,-1] = x[:, 1:], x[:, 0]
        yield x, y


###### build_inputs ##############
def build_inputs(n_seqs, n_steps):
    inputs = tf.placeholder(tf.int32, shape=[n_seqs,n_steps], name='inputs')
    targets = tf.placeholder(tf.int32, shape=[n_seqs,n_steps], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob



###### build_lstm ##############
def lstm_cell(lstm_size, keep_prob):
    LstmCell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    return tf.nn.rnn_cell.DropoutWrapper(LstmCell, output_keep_prob=keep_prob)

def build_lstm(lstm_size, n_layers, batch_size, keep_prob):
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_size, keep_prob) for _ in range(n_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state



###### build_output ##############
def build_output(lstm_out, in_size, out_size):
    seqs = tf.concat(lstm_out, 1)
    x = tf.reshape(seqs, [-1, in_size])

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name='predictions')
    return logits, out



###### build_loss ##############
def build_loss(logits, n_classes, targets):
    y_one_hot = tf.one_hot(targets, n_classes)
    y_shaped = tf.reshape(y_one_hot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_shaped)
    loss = tf.reduce_mean(loss)
    return loss


###### build_optimizer ##############
def build_optimizer(loss, learning_rate, grad_clip):
    tvars = tf.trainable_variables()
    grad,_ = tf.clip_by_global_norm(tf.gradients(loss, tvars),grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grad, tvars))
    return optimizer


###### class CharRNN ##############
class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50,
                lstm_size=128, num_layers=2,learning_rate=0.001,grad_clip=5,sampling=False):
        if sampling == True:
            batch_size, num_steps ==1,1
        else:
            batch_size, num_steps ==batch_size, num_steps

        tf.reset_default_graph()

        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)

        x_ont_hot = tf.one_hot(self.inputs, num_classes)

        outputs, state = tf.nn.dynamic_rnn(cell, x_ont_hot, initial_state=self.initial_state)
        self.final_state = state

        self.logits, self.predictions = build_output(outputs, lstm_size, num_classes)

        self.loss = build_loss(self.logits, num_classes, self.targets)

        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)



###############################################################
#########   Train   $$$$$$$$$$$$$$
#############################################
batch_size = 100
num_steps = 100
lstm_size = 512
num_layers = 1
learning_rate = 0.001
keep_prob = 0.5

epoches = 20
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)
saver = tf.train.Saver(max_to_keep=100)

###### session ##############
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    counter = 0

    for e in range(epoches):
        new_state = sess.run(model.initial_state)
        loss = 0

        for x,y in get_batches(encoded, batch_size, num_steps):
            counter+=1
            start = time.time()
            feed = {model.inputs:x,
                    model.targets:y,
                    model.keep_prob:keep_prob,
                    model.initial_state:new_state}
            batch_loss, new_state,_ = sess.run([model.loss,
                                              model.final_state,
                                              model.optimizer],
                                             feed_dict=feed)
            end = time.time()

######################  print  ###################
            if counter % 100 ==0:
                print('轮数：{}/{}...'.format(e+1,epoches),
                      '训练步数:{}...'.format(counter),
                      '训练误差：{:.4f}...'.format(batch_loss),
                      '{:.4f}sec/batch'.format((end-start)))

            if (counter % save_every_n == 0):
                saver.save(sess,'checkpoints/i{}_l{}.ckpt'.format(counter,lstm_size))
    saver.save(sess,'checkpoints/i{}_l{}.ckpt'.format(counter, lstm_size))

##### check
tf.train.get_checkpoint_state('checkpoints')


###### pick top n ##############
def pick_top_n(preds, vocab_size, top_n):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p/sum(p)

    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


###### sample ##############
def sample(checkpoint, n_samples, lstm_size, vocab_size, prime = 'The '):
    samples = [c for c in prime]
    model = CharRNN(len(vocab),lstm_size=lstm_size,sampling=True)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess,checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1,1))
            feed = {model.inputs:x,
                    model.keep_prob:1.,
                    model.initial_state:new_state}
            preds,new_state = sess.run([model.predictions,
                                        model.final_state],
                                       feed_dict=feed)
        c = pick_top_n(preds, len(vocab))

        samples.append(int_to_vocab[c])
    return ''.join(samples)


###### latest_checkpoint ##############
checkpoint = tf.train.latest_checkpoint('checkpoints')

sampp = sample(checkpoint,2000, lstm_size, len(vocab), prime = 'Far')
print(sampp)



########  born data  ##############################

checkpoint = 'checkpoints/i200_l512.ckpt'
sampp = sample(checkpoint, 1000, lstm_size,len(vocab), prime='Far')
print(sampp)

checkpoint = 'checkpoints/i2000_l512.ckpt'
sampp = sample(checkpoint, 1000, lstm_size,len(vocab), prime='Far')
print(sampp)














