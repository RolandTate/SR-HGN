import os.path as osp
import shutil
from pathlib import Path
import argparse

import numpy as np
from matplotlib import rcParams
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

import torch
from torch import nn
from torch.nn import functional as F
import time  # for tracking training and prediction time
import matplotlib.pyplot as plt  # for plotting loss
import logging

from model import SRHGN
from utils import set_random_seed, load_data, get_n_params, set_logger

# 设置matplotlib的日志级别
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def load_params():
    parser = argparse.ArgumentParser(description='Training SR-HGN')
    parser.add_argument('--prefix', type=str, default='SR-HGN')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--feat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--dataset', type=str, default='imdb')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--val_split', type=float, default=0.1)

    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--clip', type=int, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--input_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_node_heads', type=int, default=4)
    parser.add_argument('--num_type_heads', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.4)

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


def train(model, G, labels, target, optimizer, scheduler, train_idx, clip=1.0):
    model.train()

    logits, _, _ = model(G, target)
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    scheduler.step()

    return loss.item()


def eval(model, G, labels, target, train_idx, val_idx, test_idx):
    model.eval()

    start_time = time.time()

    logits, _, _ = model(G, target)
    loss = F.cross_entropy(logits[val_idx], labels[val_idx])

    end_time = time.time()
    single_prediction_time = (end_time - start_time) * 1000

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

    _, embedding, attns = model(G, target)
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

    G, node_dict, edge_dict, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, target = load_data(params['dataset'], params['train_split'], params['val_split'], params['feat'])

    G, input_dims = init_feat(G, params['input_dim'], features)
    G = G.to(device)
    labels = labels.to(device)
    print(labels.max().item() + 1 == num_classes)
    model = SRHGN(G,
                  node_dict, edge_dict,
                  input_dims=input_dims,
                  hidden_dim=params['hidden_dim'],
                  output_dim=labels.max().item() + 1,
                  num_layers=params['num_layers'],
                  num_node_heads=params['num_node_heads'],
                  num_type_heads=params['num_type_heads'],
                  alpha=params['alpha']).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=params['epochs'], max_lr=params['max_lr'])
    logger.info('Training SR-HGN with #param: {:d}'.format(get_n_params(model)))

    best_val_mif1 = 0
    best_epoch = 0

    train_losses = []
    val_losses = []

    train_micro_values = []
    val_micro_values = []

    # Start measuring training time
    start_time = time.time()

    for epoch in range(1, params['epochs'] + 1):
        loss = train(model, G, labels, target, optimizer, scheduler, train_idx, clip=params['clip'])

        train_losses.append(loss)

        if epoch % params['verbose'] == 0:
            results = eval(model, G, labels, target, train_idx, val_idx, test_idx)

            val_losses.append(results['loss'])

            train_micro_values.append(results['train_mif1'])
            val_micro_values.append(results['val_maf1'])

            if results['val_mif1'] > best_val_mif1:
                best_val_mif1 = results['val_mif1']
                best_results = results
                best_epoch = epoch

                torch.save(model.state_dict(), osp.join(checkpoints_path, f'{my_str}.pkl'))
            
            logger.info('Epoch: {:d} | LR: {:.4f} | Loss {:.4f} | Val MiF1: {:.4f} (Best: {:.4f}) | Test MiF1: {:.4f} (Best: {:.4f})'.format(
                epoch,
                optimizer.param_groups[0]['lr'],
                loss,
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

    logger.info('Best Epoch: {:d} | Train MiF1: {:.4f},  MaF1: {:.4f} | Val MiF1: {:.4f}, MaF1: {:.4f} | Test MiF1: {:.4f}, MaF1: {:.4f}'.format(
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

    fs = 28
    ls = 24
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = fs
    fig, ax1 = plt.subplots(figsize=(10, 8))

    ax1.set_xlabel('Epochs', fontsize=fs)
    ax1.set_ylabel('Loss Values', fontsize=fs, labelpad=10)

    # Set limits to start from 0 and align spines
    ax1.set_xlim(0, params['epochs'])
    ax1.set_ylim(0, max(train_losses) + 0.02 * max(train_losses))
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')

    train_loss_line, = ax1.plot(range(params['epochs']), train_losses, color='#1A2A3A', label='Loss')
    ax1.tick_params(axis='y', labelsize=ls)
    ax1.tick_params(axis='x', labelsize=ls)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Classification Micro-F1', fontsize=fs, labelpad=10)  # we already handled the x-label with ax1
    ax2.set_ylim(0, 1.02)
    ax2.spines['right'].set_position(('outward', 0))
    micro_line, = ax2.plot(range(params['epochs']), train_micro_values, color='#F25022', label='Training')
    macro_line, = ax2.plot(range(params['epochs']), val_micro_values, color='#FFB900', label='Validation')
    ax2.tick_params(axis='y', labelsize=ls)

    lines = [train_loss_line, micro_line, macro_line]
    labels = [line.get_label() for line in lines]

    # Create the legend manually
    plt.subplots_adjust(top=0.9)  # Leave space at the top for the legend

    # Create the legend manually and position it in the remaining top space
    plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=ls + 1)

    plt.show()

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
    params = load_params()
    set_random_seed(params['seed'])

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
