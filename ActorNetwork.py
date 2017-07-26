import numpy as np
import math
# from keras.initializations import normal, identity
from keras.initializers import RandomNormal
from keras.models import model_from_json
from keras.models import Sequential, Model
# from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, Lambda, Activation, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras import layers
from keras.initializers import VarianceScaling

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape=[state_size])   
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        # Steering = Dense(1,activation='tanh',init=lambda shape:VarianceScaling(scale=1e-4)(shape))(h1)  
        # Acceleration = Dense(1,activation='sigmoid',lambda shape:VarianceScaling(scale=1e-4)(shape))(h1)   
        # Brake = Dense(1,activation='sigmoid',lambda shape:VarianceScaling(scale=1e-4)(shape))(h1)
        Steering = Dense(1,activation='tanh')(h1)  
        Acceleration = Dense(1,activation='sigmoid')(h1)   
        Brake = Dense(1,activation='sigmoid')(h1)
        # V = merge([Steering,Acceleration,Brake],mode='concat')
        V = layers.concatenate([Steering,Acceleration,Brake])          
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S

class ActorNetworkMul(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        ## original version
        print("Now we build the model")
        S = Input(shape=[state_size])   
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)

        Steering = Dense(1,activation='tanh')(h1)  
        Acceleration = Dense(1,activation='sigmoid')(h1)   
        Brake = Dense(1,activation='sigmoid')(h1)
        # V = merge([Steering,Acceleration,Brake],mode='concat')
        V = layers.concatenate([Steering,Acceleration,Brake])          
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S


class ActorNetworkMulBatch(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads,
            K.learning_phase(): 1
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")

        # Batch norm version
        S = Input(shape=[state_size])
        s1 = BatchNormalization()(S)
        s1 = Dense(HIDDEN1_UNITS)(s1)
        s1 = BatchNormalization()(s1)
        s1 = Activation('relu')(s1)
        s1 = Dense(HIDDEN2_UNITS)(s1)
        s1 = BatchNormalization()(s1)
        h1 = Activation('relu')(s1)

        Steering = Dense(1,activation='tanh')(h1)  
        Acceleration = Dense(1,activation='sigmoid')(h1)   
        Brake = Dense(1,activation='sigmoid')(h1)
        # V = merge([Steering,Acceleration,Brake],mode='concat')
        V = layers.concatenate([Steering,Acceleration,Brake])          
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S

