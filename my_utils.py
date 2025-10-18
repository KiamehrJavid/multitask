import warnings
warnings.filterwarnings("ignore",message=r"Passing \(type, 1\) or '1type' as a synonym of type is deprecated",category=FutureWarning)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import argparse
import os

from train import train
from task import Trial, generate_trials
from network import Model
import tools
import json_tricks as jt


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import json


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def load_model(model_dir, verbose=False):

    if verbose: print('model_dir : '+model_dir+'\n\n')

    with open(model_dir+'/hp.json', 'r') as f:
        hp = json.load(f)

    hp['rng'] = np.random

    tf.reset_default_graph()

    model = Model(model_dir=model_dir, hp=hp, verbose=verbose)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_dir+'/model.ckpt')
    tracked_vars = json.load(open(os.path.join(model_dir, 'tracked_vars.json'),'r'))
    model.samples_seen = tracked_vars.get('samples_seen')
    model.total_train_time = tracked_vars.get('total_train_time')

    if verbose: print('Model loaded.\n')
    return model, hp, sess


def set_params(N, rule, dale, pos_win=False, data_dir='/home/kia/Desktop/PoD/Thesis/multitask/data/', debug_num = None):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ruleset = 'all'
    rule_trains = [rule]

    batch_size = 4096

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    hp = {'batch_size_train': batch_size,
            'activation': 'relu',
            'target_perf' : 1,
            'n_rnn': N,
            'mix_rule': True,
            'l1_h': 0.,
            'use_separate_input':False,
            'learnin_rate': 0.001,
            'rng':np.random}


    model_dir = data_dir + 'debug' + ('' if debug_num is None else '_'+str(debug_num))
    dale = dale # share of the inhibitory neurons
    if dale is not None:
        neur_type = np.ones(N, dtype=np.float32)
        neur_type[:int(N*dale)] *= -1
        model_dir += '/dale_'+str(dale)

        np.random.shuffle(neur_type)
        hp['neur_type'] = neur_type
    else:
        model_dir += '/nodale'

    if pos_win:
        model_dir += 'poswin_'
        hp['pos_win'] = True
    else:
        hp['pos_win'] = False
        
    rules = ''
    for rule in rule_trains:
        rules += rule + '_'
        
    model_dir += '/'+rules
    model_dir += '_N'+str(N)

    return hp, model_dir, rule_trains, ruleset


def train_varying_batch_size(model_dir, rule_trains, hp, ruleset='all',
                             niter=5, Batch_Sizes=[512,4096], max_steps=None, learning_rate = None,
                             train_from_scratch=False, verbose=True, asses = False):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    samp_ax = []
    time_ax = []
    errs = []
    cors = []
    perfs = []
    consts = []
    for iter in range(niter):
        for batch_size in Batch_Sizes:

            if ((iter == 0) and (batch_size == Batch_Sizes[0])) and train_from_scratch:
                load_dir = None
                print('Training from scratch')

            else:
                load_dir = model_dir
                print('Training continues from '+load_dir)

            display_step = hp.get('display_step') if hp.get('display_step') is not None else int(np.ceil(10*batch_size/128))
            print('display_step : '+str(display_step))
            

            if learning_rate is None:
                learning_rate = batch_size*0.0002
            # learning_rate/= (iter+1)

            hp['batch_size_train'] = batch_size
            hp['learnin_rate'] = learning_rate
            if max_steps is not None:
                hp['max_steps'] = int(max_steps)
            else:
                # hp['max_steps'] = 3e4 * np.ceil(np.log(batch_size+0.01)/np.log(4)) #2e4, 4e5
                hp['max_steps'] = np.ceil(2e4 * batch_size**0.75)

            train(model_dir,
                    load_dir=load_dir,
                    seed=17,
                    hp=hp,
                    ruleset=ruleset,
                    rule_trains=rule_trains,
                    display_step=display_step)
            

            if asses:
                tracked_vars = json.load(open(os.path.join(model_dir, 'tracked_vars.json'),'r'))
                samp_ax.append(tracked_vars.get('samples_seen'))
                time_ax.append(tracked_vars.get('total_train_time'))
                err, cor = assess_model(model_dir=model_dir, rule=rule_trains[0], plot = False, save_plots = False, verbose = False, suffix='')
                errs.append(err)
                cors.append(cor)

    if asses:
        return {'samp_ax':samp_ax, 'time_ax':time_ax, 'errs':errs, 'cors':cors}
    else:
        return None


def generate_and_predict(rule, model, hp, sess):

    trial = generate_trials(rule, hp=hp, mode='test')
    feed_dict = tools.gen_feed_dict(model, trial, hp)
    output = sess.run(model.y_hat, feed_dict=feed_dict)
    return trial, output


def Total_Absolute_Error(model=None, hp=None, model_dir=None, rule='reactgo'):

    if ((model is None) or (hp is None)) and (model_dir is None):
        print('Please provide model and hp, or model_dir')
        return None
    if model_dir is not None:
        model, hp , sess= load_model(model_dir)

    trial, output = generate_and_predict(rule, model, hp, sess)
    t_dim, n_samps, n_out_signals = output.shape
    
    err = np.abs(output - trial.y).mean()

    return err        


def pearson_over_tail(model=None, hp=None, model_dir=None, rule='reactgo', tail = 10):


    if ((model is None) or (hp is None)) and (model_dir is None):
        print('Please provide model and hp, or model_dir')
        return None
    if model_dir is not None:
        model, hp , sess= load_model(model_dir)

    trial, output = generate_and_predict(rule, model, hp, sess)
    t_dim, n_samps, n_out_signals = output.shape

    average_output_signals = np.mean(output[-tail:,:,:], axis=0)
    average_target_signals = np.mean(trial.y[-tail:,:,:], axis=0)

    corrs = [np.corrcoef(average_output_signals[i], average_target_signals[i])[0,1] for i in range(n_out_signals)]

    return np.mean(corrs)


def plot_trial(model_dir, save = False, rule='reactgo', suffix=''):
    model, hp , sess= load_model(model_dir)
    trial, output = generate_and_predict(rule=rule, model=model, hp=hp, sess=sess)

    k = np.random.randint(0, 40)
    s = np.random.randint(0, 33)
    out = output[:,k,:]
    tar = trial.y[:,k,:]



    plt.figure(figsize=(10, 3))
    plt.plot(out, label='Model Output')
    plt.title('Model output')
    plt.xlabel('Time')
    plt.ylabel('Response')
    if save:    plt.savefig(os.path.join(model_dir, 'plots', 'model_output_'+suffix+'.png'))
    plt.show()


    plt.figure(figsize=(10, 3))
    plt.plot(tar, label='Target Output', linestyle='--')
    plt.title('target output')
    plt.xlabel('Time')
    plt.ylabel('Response')
    if save:    plt.savefig(os.path.join(model_dir, 'plots', 'target_output_'+suffix+'.png'))
    plt.show()


    plt.figure(figsize=(10, 3))
    plt.plot(tar-out, label='Error')
    plt.title('Error (Target - Output)')
    plt.xlabel('Time')
    plt.ylabel('Response')
    if save:    plt.savefig(os.path.join(model_dir, 'plots', 'error_'+suffix+'.png'))
    plt.show()


    plt.figure(figsize=(5, 4))
    tail_out = out[-10:,:].mean(axis=0)
    tail_tar = tar[-10:,:].mean(axis=0)
    plt.scatter(tail_tar, tail_out, label='Tail Mean (last 10 steps)')
    plt.title('Target vs Output (Tail Mean)')
    plt.xlabel('Target')
    plt.ylabel('Model output')
    if save:    plt.savefig(os.path.join(model_dir, 'plots', 'target_vs_output_tail_'+suffix+'.png'))
    plt.show()


    plt.figure(figsize=(5, 4))
    plt.plot(tail_out, label='Tail Mean Output', marker='o')
    plt.plot(tail_tar, label='Tail Mean Target', marker='x')
    plt.title('Tail Mean Outputs (last 10 steps)')
    plt.xlabel('Output Index')
    plt.ylabel('Response')
    plt.legend()
    if save:    plt.savefig(os.path.join(model_dir, 'plots', 'tail_means_'+suffix+'.png'))
    plt.show()


def assess_model(model_dir, rule='reactgo', plot = False, save_plots = False, verbose = False, suffix=''):

    model, hp , sess= load_model(model_dir)
    err = Total_Absolute_Error(model=model, hp=hp, model_dir=model_dir, rule=rule)
    cor = pearson_over_tail(model=model, hp=hp, model_dir=model_dir, rule=rule)

    if verbose: print(f'\nFor rule {rule}, Total Absolute Error: {err}, Pearson correlation over tail: {cor}\n')

    if plot:
        plot_trial(model_dir=model_dir, save=save_plots, rule=rule, suffix=suffix)

    return err, cor


def read_res(model_dir, filename='assessment'):

    try:
        dict_dir = model_dir + '/' + filename + '.json'
        with open(dict_dir, 'r') as f:
            assessment = jt.load(f)
        print('Json file found')

    except:
        print('No json file found')
        dict_dir = model_dir + '/' + filename + '.txt'
        with open(dict_dir, 'r') as f:
            a = ''
            for line in f.readlines():
                a = a + line
            a = a.replace('np.float64(nan)', 'np.nan')
            a = a.replace('\n', '')
            a = a.replace('  ', ' ')
            a = a.replace('array', 'np.array')
            assessment = eval(a, {"np": np, "nan": np.nan})
        print('Txt file found')

    return assessment


def update_assessment(model_dir, new_assessment, dale, rule, N):

        dict_dir = model_dir + '/assessment'
        if not os.path.exists(dict_dir+'.json') and not os.path.exists(dict_dir+'.txt'):
                with open(dict_dir+'.txt', 'w') as f:
                        f.write(str(new_assessment))
                print('Saved assessment file as txt for Dale = ', dale, ' Rule = ', rule, ' N = ', N)
        else:
                prev_assessment = read_res(model_dir=model_dir, filename='assessment')
                prev_samp_ax, prev_time_ax, prev_errs, prev_cors = prev_assessment.values()
                new_samp_ax, new_time_ax, new_errs, new_cors = new_assessment.values()

                samp_ax = np.concatenate((np.array(prev_samp_ax) , np.array(new_samp_ax)))
                time_ax = np.concatenate((np.array(prev_time_ax) , np.array(new_time_ax)))
                errs = prev_errs + new_errs
                cors = prev_cors + new_cors
                assessment = {'samp_ax':samp_ax, 'time_ax':time_ax, 'errs':errs, 'cors':cors}
                try:
                        dict_dir = model_dir + '/assessment.json'
                        with open(dict_dir, 'w') as f:
                                jt.dump(assessment, f)
                        print('Saved assessment file as json for Dale = ', dale, ' Rule = ', rule, ' N = ', N)
                except:
                        dict_dir = model_dir + '/assessment.txt'
                        with open(dict_dir, 'w') as f:
                                f.write(str(assessment))
                        print('Saved assessment file as txt for Dale = ', dale, ' Rule = ', rule, ' N = ', N)
