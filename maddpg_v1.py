#%%
from flat_game import m_carmunk
import numpy as np
import random
#import argparse
import tensorflow as tf
# from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBufferMulti
from ActorNetwork import ActorNetworkMul
from CriticNetwork import CriticNetworkMul
from OU import OU
#import timeit
from optparse import OptionParser
import os


OU = OU()       #Ornstein-Uhlenbeck Process
#%%
def jsonDumper(fileName, jsonString):
    with open(fileName, 'w') as f:
        f.write(json.dumps(jsonString))

def _makeDefaltAction(numCars = 2):
    for i in xrange(numCars):
        steering = np.random.normal(0, 0.2)
        accel = random.uniform(0, 1)
        brake = random.uniform(0, 1)
        defaultAction = np.array([[steering, accel, brake]])
        if i == 0:
            actions = np.zeros((1, defaultAction.shape[1]))
            actions = defaultAction
        else:
            actions = np.vstack((actions, defaultAction))
    return actions

def stateNorm(state, width, height):
    normF = np.array([2., 2./width, 2./height, 1, 1, 1./50, 1./50, 1./50, 1./50, 1./50, 1./50, 1./50])
    sub = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    normState = state*normF-sub
    return normState

def actionAddNoise(a_t_original, train_indicator, epsilon, numCars=2):
    noise_t = np.zeros_like(a_t_original)
    a_t = np.zeros_like(a_t_original)
    for i in xrange(numCars):
        noise_t[i][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[i][0],  0.0 , 0.6, 0.2)
        noise_t[i][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[i][1],  0.5 , 1.0, 0.10)
        noise_t[i][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[i][2],  0.3 , 1.0, 0.05)

        a_t[i][0] = a_t_original[i][0] + noise_t[i][0]
        a_t[i][1] = a_t_original[i][1] + noise_t[i][1]
        a_t[i][2] = a_t_original[i][2] + noise_t[i][2]
    return a_t

def addToBuffer(buff, s_t, a_t, r_t, s_t1, done, numCars=2):
    for i in xrange(numCars):
        buff.add(s_t[i], a_t[i], r_t[i], s_t1[i], done[i])     #Add replay buffer

def addToBufferMulti(buff, s_t, a_t, r_t, s_t1, done, numCars=2):
    # make new state
    st1 = s_t1.flatten()
    st = s_t.flatten()
    s_tt = np.zeros((numCars, st1.shape[0]+a_t.shape[1]*(numCars-1)))
    s_tt1 = np.zeros((numCars, st1.shape[0]+a_t.shape[1]*(numCars-1)))
    for i in xrange(numCars):
        index = np.array([True]*numCars)
        index[i] = False
        s_tt[i] = np.hstack((st, a_t[index].flatten()))
        s_tt1[i] = np.hstack((st1, a_t[index].flatten()))

    for i in xrange(numCars):
        buff.add(s_tt[i], a_t[i], r_t[i], s_tt1[i], done[i], s_t[i], s_t1[i])     #Add replay buffer

#%%
def playGame(saveFolder, train_indicator=1, numCars=16): #1 means Train, 0 means simply Run
#%%
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperPaters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 12  #of sensors input
    
    m_state_dim = state_dim*numCars
    m_action_dim = action_dim*(numCars-1)
    critic_input_d = m_state_dim + m_action_dim

    np.random.seed(1336)

    EXPLORE = 100000.
    done = False
    step = 0
    epsilon = 1
    savePoint = 1000
    stepBreaker = 4000
    modelSavePoint = stepBreaker/40
    #%%
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
#%%
    actor = ActorNetworkMul(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetworkMul(sess, critic_input_d, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBufferMulti(BUFFER_SIZE)    #Create replay buffer
#%%
    # Create a new game instance.
    game_state = m_carmunk.GameState(numCars=numCars)
#%%
    # Get initial state by doing nothing and getting the state.
    defaultAction = _makeDefaltAction(numCars)
    s_t, _, _ = game_state.frame_step(defaultAction)
    w_ = game_state.w
    h_ = game_state.h
    s_t = stateNorm(s_t, w_, h_)

        #Now load the weight
#%%
    print("Now we load the weight")
    try:
        actor.model.load_weights(saveFolder+"/ma_actormodel.h5")
        critic.model.load_weights(saveFolder+"/ma_criticmodel.h5")
        actor.target_model.load_weights(saveFolder+"/ma_actormodel.h5")
        critic.target_model.load_weights(saveFolder+"/ma_criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("Multi-Agent Simulation Start.")
#%%
    j = 0
    total_reward = []
    losses= []
#%%
    jsonDumper(saveFolder+"/ma_actormodel.json", actor.model.to_json())
    jsonDumper(saveFolder+"/ma_criticmodel.json", critic.model.to_json())
    while True:
#        print("Step : " + str(step) + " Replay Buffer " + str(buff.count()))
        #%%
        sum_rewards = 0
        loss = 0
        epsilon -= 1.0 / EXPLORE

        #%%
        a_t_original = actor.model.predict(s_t)
        #%%
        a_t = actionAddNoise(a_t_original, train_indicator, epsilon, numCars=numCars)
        s_t1, r_t, done = game_state.frame_step(a_t)
        s_t1 = stateNorm(s_t1, w_, h_)
        #%%
        addToBufferMulti(buff, s_t, a_t, r_t, s_t1, done, numCars=numCars)

        #Do the batch update
        batch = buff.getBatch(BATCH_SIZE)
#        buff.add(s_tt[i], a_t[i], r_t[i], s_tt1[i], done[i], s_t[i], s_t1[i]) 
#%%
        states_critic = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states_critic = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        states = np.asarray([e[5] for e in batch])
        new_states = np.asarray([e[6] for e in batch])
        y_t = np.asarray([e[1] for e in batch])
#%%
        target_q_values = critic.target_model.predict([new_states_critic, actor.target_model.predict(new_states)])  
       #%%
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA*target_q_values[k]
    #%%
        if (train_indicator):
            loss = critic.model.train_on_batch([states_critic,actions], y_t)
            a_for_grad = actor.model.predict(states)
            grads = critic.gradients(states_critic, a_for_grad)
            actor.train(states, grads)
            actor.target_train()
            critic.target_train()
#%%
        s_t = s_t1
        j += 1

        sum_rewards+= r_t.sum()
        if np.mod(j, savePoint) == 0:
            step += 1
            if (train_indicator):
                print("Now we save model : {}".format(step))
                actor.model.save_weights(saveFolder+"/ma_actormodel.h5", overwrite=True)
                jsonDumper(saveFolder+"/ma_actormodel.json", actor.model.to_json())
                critic.model.save_weights(saveFolder+"/ma_criticmodel.h5", overwrite=True)
                jsonDumper(saveFolder+"/ma_criticmodel.json", critic.model.to_json())
                losses.append(loss)
                total_reward.append(sum_rewards/j)
                jsonDumper(saveFolder+"/ma_losses.json", str(losses))
                jsonDumper(saveFolder+"/ma_rewards.json", str(total_reward))
                print("Loss of Critic : ", loss)
                print("Avg Rewards : ", sum_rewards/j)
                if np.mod(step, modelSavePoint) == 0:
                    actor.model.save_weights(saveFolder+"/ma_actormodel_{}.h5".format(step), overwrite=True)
                    critic.model.save_weights(saveFolder+"/ma_criticmodel_{}.h5".format(step), overwrite=True)
                    jsonDumper(saveFolder+"/ma_losses_{}.json".format(step), str(losses))
                    jsonDumper(saveFolder+"/ma_rewards_{}.json".format(step), str(total_reward))
            j = 0
            if step == stepBreaker:
                break


#    print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
    print("Total Step: " + str(step))
    print("")
    print("Finish.")

if __name__ == "__main__":
    use = "Usage : %prog [option]"
    parser = OptionParser(usage=use)
    parser.add_option("-t", "--test", dest="test", action="store_true", 
        default=False, help="set if you want to train model")
    parser.add_option("-n", "--num", dest="n_cars", type="int", default=16,
        help="Number of Cars")
    parser.add_option("-s", "--save", dest="save", default="Dropbox/06.MLJeju/MADDPG/results/ma2",
        help="Model save directory")


    options, args = parser.parse_args()
    saveFolder = os.path.expanduser(os.path.join("~",options.save))
    
        
    if options.test: train = 0
    else : 
        train = 1
        if not os.path.isdir(saveFolder):
            print("Folder's are not exist so make it")
            os.makedirs(saveFolder)
    # saveHome = os.path.expandusers()
    # saveHome = options.savePoint
    train_indicator=train
    numCars=options.n_cars
    playGame(saveFolder, train_indicator=train_indicator, numCars=numCars)






