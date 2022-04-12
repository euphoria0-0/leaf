"""Script to run the baselines."""
import argparse
from cmath import phase
import importlib
from turtle import pos
import numpy as np
import os
import sys
import random
import time
import tensorflow as tf

import metrics.writer as metrics_writer

from utils.baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from models.model import ServerModel
from client_selection import *

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

def main():

    args = parse_args()

    args.start = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    args.save_path = f'{args.save_path}/{args.dataset}_{args.t[0]}/{args.metrics_name}_{args.start}'
    os.makedirs(args.save_path, exist_ok=True)

    opts_file = open(f'{args.save_path}/options.txt', 'w')
    for arg in vars(args):
        opts_file.write(f'{arg} = {getattr(args, arg)}\n')
    opts_file.close()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    model_path = '%s/%s/%s.py' % ('models', args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s.%s' % ('models', args.dataset, args.model)
    
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path, 'models')
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path.replace('models.','')]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(args.seed, *model_params)

    # Create server
    server = Server(client_model)

    # Create clients
    clients = setup_clients(args.dataset, client_model, args.use_val_set, args.dataset_path)
    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))

    # Initial status
    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

    # client selection method
    client_selection = getattr(sys.modules[__name__], args.method)(
        n_samples=client_num_samples,
        num_clients=clients_per_round
    )
    if args.method in LOSS_BASED_SELECTION:
        client_selection.set_hyperparams(args)
    if args.method in CLUSTERED_SAMPLING:
        client_selection.set_client_ids([c.id for c in clients])
    server.set_client_selection_method(client_selection)

    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round if args.num_available is None else args.num_available))
        
        # buffer client
        online_clients = online(clients, i, args.num_available)

        # (PRE) Select clients to train this round
        if args.method in LOSS_BASED_SELECTION:
            server.set_possible_clients(online_clients)  # just set available clients to measure metrics
        else:
            server.select_clients(i, online_clients, num_clients=clients_per_round)

        # Simulate server model training on selected clients' data
        sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch)
        
        # (POST) Select clients to train this round
        if args.method in LOSS_BASED_SELECTION:
            # measure train loss
            train_stat_metrics = server.test_model(online_clients, set_to_use='train')
            train_losses = [train_stat_metrics[c]['loss'] for c in sorted(train_stat_metrics)]
            
            server.select_clients(i, online_clients, num_clients=clients_per_round, metric=train_losses)
        
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)
        sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)
        

        # Update server model
        server.update_model()

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
    
    # Save server model
    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, '{}_{}.ckpt'.format(args.model, args.metric_name)))
    print('Model saved in path: %s' % save_path)

    # Close models
    server.close_model()



def online(clients, round, num_available=None):
    """We assume all users are always online."""
    """I assume only subset of users are online."""
    if num_available is not None:
        num_clients = min(len(clients), num_available)
        np.random.seed(round)
        selected_clients = np.random.choice(clients, num_clients, replace=False)
    else:
        selected_clients = clients
    return selected_clients


def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients


def setup_clients(dataset, model=None, use_val_set=False, data_path='..'):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join(data_path, 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join(data_path, 'data', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, groups, train_data, test_data, model)

    return clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.save_path, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.save_path, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn


def print_stats(
    num_round, server, clients, num_samples, args, writer, use_val_set):
    
    train_stat_metrics = server.test_model(clients, set_to_use='train')
    print_metrics(train_stat_metrics, num_samples, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


if __name__ == '__main__':
    main()
