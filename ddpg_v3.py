#%%
from flat_game import m_carmunk
# import numpy as np
# import random
#from keras.models import model_from_json
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.optimizers import Adam
import tensorflow as tf
# from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

#import timeit
from optparse import OptionParser
from util import *
import os



#%%
def playGame(saveFolder, train_indicator=1, numCars=16): #1 means Train, 0 means simply Run
#%%
    if train_indicator==0: test = True
    else : test=False
    BUFFER_SIZE = 100000
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 12  #of sensors input

    # np.random.seed(1337)

    EXPLORE = 100000.
    done = False
    step = 0
    epsilon = 1
    savePoint = 1000
    stepBreaker = 400
    modelSavePoint = stepBreaker/40
    #%%
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
#%%
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
#%%
    # Create a new game instance.
    game_state = m_carmunk.GameState(numCars=numCars, test=test)
#%%
    # Get initial state by doing nothing and getting the state.
    defaultAction = makeDefaltAction(numCars)
    s_t, _, _ = game_state.frame_step(defaultAction)
    w_ = game_state.w
    h_ = game_state.h
    s_t = stateNorm2(s_t, w_, h_)
        #Now load the weight
#%%
    print("Now we load the weight")
    try:
        actor.model.load_weights(saveFolder+"actormodel_90.h5")
        critic.model.load_weights(saveFolder+"criticmodel_90.h5")
        actor.target_model.load_weights(saveFolder+"actormodel_90.h5")
        critic.target_model.load_weights(saveFolder+"criticmodel_90.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("Single-Agents Simulation Start.")
#%%
    j = 0
    total_reward = []
    losses= []
#%%
    jsonDumper(saveFolder+"/actormodel.json", actor.model.to_json())
    jsonDumper(saveFolder+"/criticmodel.json", critic.model.to_json())
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
        s_t1 = stateNorm2(s_t1, w_, h_)
        #%%
        addToBuffer(buff, s_t, a_t, r_t, s_t1, done, numCars=numCars)
        # buff.add(s_t, a_t, r_t, s_t1, done)     #Add replay buffer

        #Do the batch update
        batch = buff.getBatch(BATCH_SIZE)
        #%%
        # print(batch) # batch is list object
        # print(len(batch))
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
       #%%
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA*target_q_values[k]
    #%%
        if (train_indicator):
            loss = critic.model.train_on_batch([states,actions], y_t)
            a_for_grad = actor.model.predict(states)
            grads = critic.gradients(states, a_for_grad)
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
                actor.model.save_weights(saveFolder+"/actormodel.h5", overwrite=True)
                jsonDumper(saveFolder+"/actormodel.json", actor.model.to_json())
                critic.model.save_weights(saveFolder+"/criticmodel.h5", overwrite=True)
                jsonDumper(saveFolder+"/criticmodel.json", critic.model.to_json())
                losses.append(loss)
                total_reward.append(sum_rewards/j)
                jsonDumper(saveFolder+"/losses.json", str(losses))
                jsonDumper(saveFolder+"/rewards.json", str(total_reward))
                print("Loss of Critic : ", loss)
                print("Avg Rewards : ", sum_rewards/j)
                if np.mod(step, modelSavePoint) == 0:
                    actor.model.save_weights(saveFolder+"/actormodel_{}.h5".format(step), overwrite=True)
                    jsonDumper(saveFolder+"/actormodel.json", actor.model.to_json())
                    critic.model.save_weights(saveFolder+"/criticmodel_{}.h5".format(step), overwrite=True)
                    jsonDumper(saveFolder+"/criticmodel.json", critic.model.to_json())
                    losses.append(loss)
                    jsonDumper(saveFolder+"/losses_{}.json".format(step), str(losses))
                    jsonDumper(saveFolder+"/rewards_{}.json".format(step), str(total_reward))
            j = 0
            if step == stepBreaker:
                break

                


#    print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
    print("Total Step: " + str(step))
    print("")
    print("Finish.")

if __name__ == "__main__":
    #%%
    use = "Usage : %prog [option]"
    parser = OptionParser(usage=use)
    parser.add_option("-t", "--test", dest="test", action="store_true", 
        default=False, help="set if you want to train model")
    parser.add_option("-n", "--num", dest="n_cars", type="int", default=16,
        help="Number of Cars")
    parser.add_option("-s", "--save", dest="save", default="Dropbox/06.MLJeju/MADDPG/results",
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
    #%%
    playGame(saveFolder, train_indicator=train_indicator, numCars=numCars)






