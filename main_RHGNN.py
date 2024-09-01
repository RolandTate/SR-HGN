import os.path as osp
import shutil
import warnings
from pathlib import Path
import argparse
import copy
import time  # for tracking training and prediction time
import matplotlib.pyplot as plt  # for plotting loss
import logging

import numpy as np
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

import torch
from torch import nn
from torch.nn import functional as F

from model_HGT.HGT import HGT
from model_RHGNN.R_HGNN import R_HGNN
from model_y_v3.Classifier import Classifier
from utils import set_random_seed, load_data, get_n_params, set_logger
from torchinfo import summary

# 设置matplotlib的日志级别
logging.getLogger('matplotlib').setLevel(logging.WARNING)

args = {
    'dataset': 'OGB_MAG',
    'model_name': 'R_HGT_lr0.001_dropout0.5_seed_0_patience20',
    'predict_category': 'paper',
    'seed': 0,
    'cuda': 0,
    'learning_rate': 0.001,
    'num_heads': 8,
    'hidden_units': 64,
    'relation_hidden_units': 8,
    'dropout': 0.5,
    'n_layers': 2,
    'residual': True,
    'batch_size': 1280,  # the number of nodes to train in each batch
    'node_neighbors_min_num': 10,  # number of sampled edges for each type for each GNN layer
    'optimizer': 'adam',
    'weight_decay': 0.0,
    'epochs': 200,
    'patience': 20
}

def load_params():
    parser = argparse.ArgumentParser(description='Training R_HGNN')
    parser.add_argument('--prefix', type=str, default='R_HGNN')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--feat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='imdb')  # acm, dblp, imdb
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--val_split', type=float, default=0.1)

    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--clip', type=int, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--relation_hidden_dim', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--residual', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--cluster', action='store_false')

    args = parser.parse_args()
    args = vars(args)

    return args


def init_feat(G, n_inp, features):
    # Randomly initialize features if features don't exist
    input_dims = {}

    for ntype in G.ntypes:
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), n_inp), requires_grad=True)
        nn.init.xavier_uniform_(emb)

        feats = features.get(ntype, emb)
        G.nodes[ntype].data['x'] = feats
        input_dims[ntype] = feats.shape[1]

    return G, input_dims


def train(model, G, labels, target, optimizer, scheduler, train_idx, clip=1.0, alpha=1.0):
    model.train()

    # input_features = {(stype, etype, dtype): G.srcnodes[dtype].data['x'] for stype, etype, dtype in
    #                   G.canonical_etypes}
    # nodes_representation, _ = model[0](G, input_features)
    nodes_representation, _ = model[0](G)

    logits = model[1](nodes_representation[target])

    loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    # loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    scheduler.step()

    return loss.item()


def eval(model, G, labels, target, train_idx, val_idx, test_idx, alpha):
    model.eval()

    start_time = time.time()

    # input_features = {(stype, etype, dtype): G.srcnodes[dtype].data['x'] for stype, etype, dtype in
    #                   G.canonical_etypes}
    # nodes_representation, _ = model[0](G, input_features)
    nodes_representation, _ = model[0](G)

    logits = model[1](nodes_representation[target])

    end_time = time.time()
    single_prediction_time = (end_time - start_time) * 1000

    loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    # loss = F.cross_entropy(logits[val_idx], labels[val_idx])

    pred = logits.argmax(1).detach().cpu().numpy()

    train_macro_f1 = f1_score(labels[train_idx].cpu(), pred[train_idx], average='macro')
    train_micro_f1 = f1_score(labels[train_idx].cpu(), pred[train_idx], average='micro')
    val_macro_f1 = f1_score(labels[val_idx].cpu(), pred[val_idx], average='macro')
    val_micro_f1 = f1_score(labels[val_idx].cpu(), pred[val_idx], average='micro')
    test_macro_f1 = f1_score(labels[test_idx].cpu(), pred[test_idx], average='macro')
    test_micro_f1 = f1_score(labels[test_idx].cpu(), pred[test_idx], average='micro')

    return {
        'train_maf1': train_macro_f1,
        'train_mif1': train_micro_f1,
        'val_maf1': val_macro_f1,
        'val_mif1': val_micro_f1,
        'test_maf1': test_macro_f1,
        'test_mif1': test_micro_f1,
        'loss': loss.item(),
        'SPT': single_prediction_time
    }


def cluster(model, G, target, labels):
    model.eval()

    # input_features = {(stype, etype, dtype): G.srcnodes[dtype].data['x'] for stype, etype, dtype in
    #                   G.canonical_etypes}
    # nodes_representation, _ = model[0](G, input_features)
    nodes_representation, _ = model[0](G)
    embedding = nodes_representation[target]

    embedding = embedding.detach().cpu().numpy()
    labels = labels.cpu()

    kmeans = KMeans(n_clusters=len(torch.unique(labels)), random_state=42).fit(embedding)
    nmi = normalized_mutual_info_score(labels, kmeans.labels_)
    ari = adjusted_rand_score(labels, kmeans.labels_)

    return {
        'nmi': nmi,
        'ari': ari
    }


def main(params):
    device = torch.device(f"cuda:{params['gpu']}" if torch.cuda.is_available() else 'cpu')

    my_str = f"{params['prefix']}_{params['dataset']}"

    logger = set_logger(my_str)
    logger.info(params)

    checkpoints_path = f'checkpoints'
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)

    G, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target = load_data(
        params['dataset'], params['train_split'], params['val_split'], params['feat'])

    G, input_dims = init_feat(G, params['input_dim'], features)
    G = G.to(device)
    labels = labels.to(device)

    # r_hgnn = R_HGNN(graph=G,
    #                 input_dim_dict=input_dims,
    #                 hidden_dim=args['hidden_units'], relation_input_dim=args['relation_hidden_units'],
    #                 relation_hidden_dim=args['relation_hidden_units'],
    #                 num_layers=args['n_layers'], n_heads=args['num_heads'], dropout=args['dropout'],
    #                 residual=args['residual'])
    #
    # classifier = Classifier(n_hid=args['hidden_units'] * args['num_heads'], n_out=num_classes)
    # model = nn.Sequential(r_hgnn, classifier).to(device)

    hgt = HGT(
        in_dims=input_dims,
        hidden_dim=64, out_dim=64, num_heads=8, ntypes=G.ntypes, etypes=G.canonical_etypes,
        num_layers=2, dropout=0.1)
    classifier = Classifier(n_hid=args['hidden_units'], n_out=num_classes)
    model = nn.Sequential(hgt, classifier).to(device)

    # mshgae = MSHGAE(graph=G,
    #               input_dim_dict=input_dims,
    #               hidden_dim=params['hidden_dim'], relation_input_dim=params['relation_hidden_dim'],
    #               relation_hidden_dim=params['relation_hidden_dim'],
    #               num_layers=params['num_layers'], n_heads=params['num_heads'], dropout=params['dropout'],
    #               residual=params['residual'])
    #
    # classifier = Classifier(n_hid=params['hidden_dim'] * params['num_heads'], n_out=num_classes)
    #
    # model = nn.Sequential(mshgae, classifier).to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=params['epochs'], max_lr=params['max_lr'])
    logger.info('Training MSHGAE with #param: {:d}'.format(get_n_params(model)))

    best_val_mif1 = 0
    best_epoch = 0

    train_losses = []
    val_losses = []

    train_micro_values = []
    val_micro_values = []

    # Start measuring training time
    start_time = time.time()

    for epoch in range(1, params['epochs'] + 1):
        loss = train(model, G, labels, target, optimizer, scheduler, train_idx, clip=params['clip'], alpha=params['alpha'])

        train_losses.append(loss)


        results = eval(model, G, labels, target, train_idx, val_idx, test_idx, params['alpha'])

        val_losses.append(results['loss'])

        train_micro_values.append(results['train_mif1'])
        val_micro_values.append(results['val_maf1'])

        if results['val_mif1'] > best_val_mif1:
            best_val_mif1 = results['val_mif1']
            best_results = results
            best_epoch = epoch

            torch.save(model.state_dict(), osp.join(checkpoints_path, f'{my_str}.pkl'))

        logger.info(
            'Epoch: {:d} | LR: {:.4f} | Loss {:.4f} | Train MiF1 {:.4f} | Val MiF1: {:.4f} (Best: {:.4f}) | Test MiF1: {:.4f} (Best: {:.4f})'.format(
                epoch,
                optimizer.param_groups[0]['lr'],
                loss,
                results['train_mif1'],
                results['val_mif1'],
                best_results['val_mif1'],
                results['test_mif1'],
                best_results['test_mif1']
             ))

            # torch.save(model.state_dict(), osp.join(checkpoints_path, f'{my_str}_{epoch}.pkl'))

        torch.cuda.empty_cache()

    # End measuring training time
    end_time = time.time()
    training_time = end_time - start_time

    logger.info(
        'Best Epoch: {:d} | Train MiF1: {:.4f},  MaF1: {:.4f} | Val MiF1: {:.4f}, MaF1: {:.4f} | Test MiF1: {:.4f}, MaF1: {:.4f}'.format(
            best_epoch,
            best_results['train_mif1'],
            best_results['train_maf1'],
            best_results['val_mif1'],
            best_results['val_maf1'],
            best_results['test_mif1'],
            best_results['test_maf1']
        ))

    if params['cluster']:
        # model.load_state_dict(torch.load(osp.join(checkpoints_path, f'{my_str}_{best_epoch}.pkl')))
        model.load_state_dict(torch.load(osp.join(checkpoints_path, f'{my_str}.pkl')))
        cluster_results = cluster(model, G, target, labels)

        logger.info('NMI: {:.4f} | ARI: {:.4f}'.format(cluster_results['nmi'], cluster_results['ari']))

    # fig, ax1 = plt.subplots()
    #
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss Values', color='tab:blue')
    # train_loss_line, = ax1.plot(range(params['epochs']), train_losses, color='tab:blue', label='train_loss values')
    # val_loss_line, = ax1.plot(range(params['epochs']), val_losses, color='tab:orange', label='val_loss values')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')
    #
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Classification Micro-F1', color='tab:red')  # we already handled the x-label with ax1
    # ax2.set_ylim(0, 1.05)
    # micro_line, = ax2.plot(range(params['epochs']), train_micro_values, color='tab:green', label='Training Micro-F1')
    # macro_line, = ax2.plot(range(params['epochs']), val_micro_values, color='tab:red', label='Validating Macro-F1')
    # ax2.tick_params(axis='y', labelcolor='tab:red')
    #
    # # fig.tight_layout(pad=3.0)  # otherwise the right y-label is slightly clipped
    # # plt.title('Training Loss and Metrics')
    #
    # # Collect handles and labels from both axes
    # lines = [train_loss_line, val_loss_line, micro_line, macro_line]
    # labels = [line.get_label() for line in lines]
    #
    # # Create the legend manually
    # ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    # plt.show()

    # Print model parameters
    model_params = get_n_params(model)
    print(f'Model Parameters: {model_params}')

    # Measure and print model size
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / 1024 ** 2  # in MB
    print(f'Model Size: {model_size:.2f} MB')

    time_per_epoch = training_time / params['epochs']
    print(f"Time per epoch: {time_per_epoch:.2f} seconds")

    print(f"Single Prediction Time: {best_results['SPT']:.2f} msec")

    return best_results, cluster_results


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    params = load_params()
    set_random_seed(params['seed'])
    # main(params)

    micro_f1_scores = []
    macro_f1_scores = []
    nmi_scores = []
    ari_scores = []
    num_runs = 10

    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}")
        results, cluster_results = main(params)
        micro_f1_scores.append(results['test_mif1'])
        macro_f1_scores.append(results['test_maf1'])
        nmi_scores.append(cluster_results['nmi'])
        ari_scores.append(cluster_results['ari'])

    avg_micro_f1 = np.mean(micro_f1_scores)
    std_micro_f1 = np.std(micro_f1_scores)
    avg_macro_f1 = np.mean(macro_f1_scores)
    std_macro_f1 = np.std(macro_f1_scores)
    avg_nmi = np.mean(nmi_scores)
    std_nmi = np.std(nmi_scores)
    avg_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)

    # 输出10次运行的详细值在一行
    micro_f1_scores_str = ', '.join([f"{score * 100:.2f}" for score in micro_f1_scores])
    print(f"Micro-F1 Scores for each run: [{micro_f1_scores_str}]")
    print(f"Average Best Micro-F1 Score over {num_runs} runs: {avg_micro_f1 * 100:.2f} ({std_micro_f1 * 100:.2f})")

    macro_f1_scores_str = ', '.join([f"{score * 100:.2f}" for score in macro_f1_scores])
    print(f"Macro-F1 Scores for each run: [{macro_f1_scores_str}]")
    print(f"Average Best Macro-F1 Score over {num_runs} runs: {avg_macro_f1 * 100:.2f} ({std_macro_f1 * 100:.2f})")

    nmi_scores_str = ', '.join([f"{score * 100:.2f}" for score in nmi_scores])
    print(f"NMI Scores for each run: [{nmi_scores_str}]")
    print(f"Average NMI Score over {num_runs} runs: {avg_nmi * 100:.2f} ({std_nmi * 100:.2f})")

    ari_scores_str = ', '.join([f"{score * 100:.2f}" for score in ari_scores])
    print(f"ARI Scores for each run: [{ari_scores_str}]")
    print(f"Average ARI Score over {num_runs} runs: {avg_ari * 100:.2f} ({std_ari * 100:.2f})")

