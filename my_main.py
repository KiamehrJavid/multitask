import warnings
warnings.filterwarnings("ignore",message=r"Passing \(type, 1\) or '1type' as a synonym of type is deprecated",category=FutureWarning)



import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import argparse
import os
from train import train
from task import Trial
import numpy as np



# print('\n\n\n\n')
# print('\n\n\n\n')

# this is the list of every task implemented already
# ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
#               'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
#               'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
#               'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']



num_ring = 2
n_rule = 1
n_eachring = 32
n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
ruleset = 'all'
rule_trains = ['reactgo']

my_hp = {
            # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 512,
            # input type: normal, multi
            'in_type': 'normal',
            # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
            'rnn_type': 'LeakyRNN',
            # whether rule and stimulus inputs are represented separately
            'use_separate_input': False,
            # Type of loss functions
            'loss_type': 'lsq',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation runctions, relu, softplus, tanh, elu
            'activation': 'relu',
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 20,
            # discretization time step/time constant
            'alpha': 0.2,
            # recurrent noise
            'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
            'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 0,
            # l2 regularization on activity
            'l2_h': 0,
            # l2 regularization on weight
            'l1_weight': 0,
            # l2 regularization on weight
            'l2_weight': 0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0,
            # proportion of weights to train, None or float between (0, 1)
            'p_weight_train': None,
            # Stopping performance
            'target_perf': 1.,
            # number of units each ring
            'n_eachring': n_eachring,
            # number of rings
            'num_ring': num_ring,
            # number of rules
            'n_rule': n_rule,
            # first input index for rule units
            'rule_start': 1+num_ring*n_eachring,
            # number of input units
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            # number of recurrent units
            'n_rnn': 256,
            # number of input units
            'ruleset': ruleset,
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.001,
            # intelligent synapses parameters, tuple (c, ksi)
            'c_intsyn': 0,
            'ksi_intsyn': 0,
            }
my_hp['rng'] = np.random.RandomState(np.random.randint(0, 1e9))
config = my_hp.copy()

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--modeldir', type=str, default='data/debug')
args = parser.parse_args()

batch_size = 256
display_step = np.ceil(100*64/batch_size)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
hp = {'batch_size_train': batch_size,
        'activation': 'softplus',
        'target_perf' : 1 - 1e-6,
        'n_rnn': 64,
        'mix_rule': True,
        'l1_h': 0.,
        'use_separate_input':False}

train(args.modeldir,
        seed=17,
        hp=hp,
        ruleset=ruleset,
        rule_trains=rule_trains,
        display_step=display_step)




### playground


# code to write a sample trial

def write_sample_trial():

        dt = config['dt']
        rng = config['rng']
        batch_size = 1
        anti_response = False

        # # each batch consists of sequences of equal length
        # # A list of locations of fixation points and fixation off time
        stim_ons = int(rng.uniform(500,2500)/dt)
        tdim = int(500/dt) + stim_ons

        # # A list of locations of stimuluss (they are always on)
        stim_locs = rng.uniform(0, 2*np.pi, (batch_size,))

        stim_mod  = rng.choice([1,2])


        '''elif mode == 'test':
        tdim = int(2500/dt)
        n_stim_loc, n_stim_mod = batch_shape = 20, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)

        stim_ons  = int(2000/dt)
        stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
        stim_mod   = ind_stim_mod + 1

        elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        batch_size = len(stim_locs)

        # Time of stimuluss on/off
        stim_ons = int(1000/dt)
        tdim = int(400/dt) + stim_ons
        stim_mod   = 1

        else:
        raise ValueError('Unknown mode: ' + str(mode))'''

        # # time to check the saccade location
        check_ons  = stim_ons + int(100/dt)

        # # Response locations
        stim_locs = np.array(stim_locs)
        if not anti_response:
                response_locs = stim_locs
        else:
                response_locs = (stim_locs+np.pi)%(2*np.pi)




        trial = Trial(config, tdim, batch_size)
        trial.add('fix_in')
        trial.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
        trial.add('fix_out', offs=stim_ons)
        trial.add('out', response_locs, ons=stim_ons)
        trial.add_c_mask(pre_offs=stim_ons, post_ons=check_ons)

        # trial.epochs = {'fix1'     : (None, stim_ons),
        #                 'go1'      : (stim_ons, None)}

        print('\n\n\n')
        print(dir(trial))
        print('\n\n\n')
        print(np.sum(trial.x))
        print('\n\n\n')
        print(np.sum(trial.y))


        print('\n\n\n')
        print(trial.x.shape)
        print('\n\n\n')
        print(trial.y.shape)

        X = trial.x
        X = X.reshape((X.shape[0], -1))
        Y = trial.y
        Y = Y.reshape((Y.shape[0], -1))
        np.savetxt(args.modeldir+'/trial_x.txt', X)
        np.savetxt(args.modeldir+'/trial_y.txt', Y)

# write_sample_trial()

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# import sys
# print(sys.version)