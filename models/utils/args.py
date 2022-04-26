import argparse

DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']
SIM_TIMES = ['small', 'medium', 'large']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    required=True)
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=1)
    parser.add_argument('-A','--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('-B','--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
    parser.add_argument('--metrics-name', 
                    help='name for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    """parser.add_argument('--metrics-dir', 
                    help='dir for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)"""
    parser.add_argument('--use-val-set', 
                    help='use validation set;', 
                    action='store_true')

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    epoch_capability_group.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)

    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='large')
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)

    # config
    parser.add_argument('--method',
                    help='client selection method;',
                    type=str,
                    default='RandomSelection',
                    required=False)
    parser.add_argument('--dataset_path',
                    help='dataset path;',
                    type=str,
                    default='..',
                    required=False)
    parser.add_argument('--save-path',
                    help='path to save results;',
                    type=str,
                    default='results/',
                    required=False)
    parser.add_argument('--save_probs', 
                        help='save values',
                        action='store_true', 
                        default=False)
    parser.add_argument('--alpha',
                        help='alpha for value function in loss-based sampling methods',
                        type=float,
                        default=1)
    parser.add_argument('-n','--num_available',
                        help='number of available clients;',
                        type=int,
                        default=None)
    parser.add_argument('--buffer_size',
                        help='number of buffer clients;',
                        type=int,
                        default=None)
    parser.add_argument('--loss',
                    help='avg/total/sqrt loss;',
                    type=str,
                    default='avg',
                    required=False)
    


    return parser.parse_args()
