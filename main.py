import os
import sys
import warnings
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import vstack, load_npz
from tqdm import tqdm
from datetime import datetime
from logzero import logger, logfile
from xclib.data import data_utils
from functools import partial

from modules import XMLDataset, MLLinear
from models import Model, test_baseline
from configs import get_configs
from evaluation import *


def load_data(dataset, baseline=None, arg1='train', arg2='label'):
    if dataset in ['eurlex', 'wiki10', 'amazon670k']:
        mode = {'train': 'trn', 'test': 'tst'}[arg1]
        if arg2 == 'feature': return data_utils.read_sparse_file(f'data/{dataset}/{mode}_X_Xf.txt')
        elif arg2 == 'label': return data_utils.read_sparse_file(f'data/{dataset}/{mode}_X_Y.txt').astype('int32')
        elif arg2 == 'score': return data_utils.read_sparse_file(f'data/{dataset}/{baseline}/{mode}_SC.txt')
    elif dataset.endswith('xtransformer'):
        mode = {'train': 'trn', 'test': 'tst'}[arg1]
        if arg2 == 'feature': return load_npz(f'data/{dataset}/X.{mode}.npz')
        elif arg2 == 'label': return load_npz(f'data/{dataset}/Y.{mode}.npz').astype('int32')
        elif arg2 == 'score': return load_npz(f'data/{dataset}/{baseline}/{mode}.pred.npz')
    else:
        assert False, 'Unknown dataset'

load_train_features = partial(load_data, arg1='train', arg2='feature')
load_train_labels = partial(load_data, arg1='train', arg2='label')
load_train_scores = partial(load_data, arg1='train', arg2='score')
load_test_features = partial(load_data, arg1='test', arg2='feature')
load_test_labels = partial(load_data, arg1='test', arg2='label')
load_test_scores = partial(load_data, arg1='test', arg2='score')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser('RDE')
    parser.add_argument('--dataset', '-d', default='eurlex', type=str)
    parser.add_argument('--baseline', '-b', default='parabel', type=str)

    parser.add_argument('--num_experts', default=3, type=int)
    parser.add_argument('--output_mode', default='residual', type=str)
    parser.add_argument('--use_norm', default=True, type=bool)
    parser.add_argument('--drop_prob', default=0.7, type=float)
    parser.add_argument('--warm_up', default=5, type=int)
    parser.add_argument('--div_factor', default=1, type=float)

    parser.add_argument('--test-baseline', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--tag', default=None, type=str)

    args = parser.parse_args()

    dataset = args.dataset
    baseline = args.baseline
    if baseline == 'xtransformer' and not dataset.endswith('xtransformer'):
        dataset = f'{dataset}-xtransformer'

    cfg = get_configs(dataset, baseline)
    ### update config
    cfg['model']['num_experts'] = args.num_experts
    cfg['model']['output_mode'] = args.output_mode
    cfg['model']['use_norm'] = args.use_norm
    cfg['model']['drop_prob'] = args.drop_prob
    cfg['model']['warm_up'] = args.warm_up
    cfg['train']['div_factor'] = args.div_factor

    model_dir = f'models/{dataset}/{baseline}'
    log_dir = f'logs/{dataset}/{baseline}'

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_name = f'log_{timestamp}'
    if args.tag is not None:
        log_name += f'_{args.tag}'
    logfile(f'{log_dir}/{log_name}.txt')
    logger.info(cfg)

    train_labels = load_train_labels(dataset)
    inv_w = get_inv_propensity(train_labels, cfg['a'], cfg['b'])
    mlb = MultiLabelBinarizer(range(train_labels.shape[1]), sparse_output=True)
    mlb.fit(None)
        
    test_features = load_test_features(dataset)
    test_labels = load_test_labels(dataset)
    test_scores = load_test_scores(dataset, baseline)
    test_loader = DataLoader(XMLDataset(test_features, test_scores, test_labels, training=False),
                             batch_size=cfg['test']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
    
    if args.test_baseline:
        test_baseline(test_loader, inv_w, mlb, logger)
        
    if args.train or args.test:
        model = Model(network=MLLinear, **cfg['data'], **cfg['model'])
    
    if args.train:
        train_features = load_train_features(dataset)
        train_scores = load_train_scores(dataset, baseline)
        train_loader = DataLoader(XMLDataset(train_features, train_scores, train_labels, training=True),
                                  batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        label_freq = np.asarray(train_labels.sum(axis=0)).squeeze()
        
        if args.reload:
            model.load_model(model_dir)
        model.train(train_loader, test_loader, label_freq, inv_w, mlb, logger, model_dir, **cfg['data'], **cfg['train'])
        
    if args.test:
        model.test(test_loader, inv_w, mlb, logger, model_dir, **cfg['data'], **cfg['train'])
    
    