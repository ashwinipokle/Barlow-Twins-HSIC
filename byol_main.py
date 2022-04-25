from byol import BYOL
import argparse
import os
import utils
from utils import str2bool

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile, clever_format
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_tuple in train_bar:
        (pos_1, pos_2), _ = data_tuple
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

        loss = net(pos_1, pos_2)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} bsz:{} f_dim:{} proj_hidden_sim: {} dataset: {}'.format(\
                                epoch, epochs, total_loss / total_num, batch_size, args.feature_dim, args.proj_hidden_dim, args.dataset))
    return total_loss / total_num

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank and target bank
        for data_tuple in tqdm(memory_data_loader, desc='Feature extracting'):
            (data1, data2), target = data_tuple
            target_bank.append(target)
            feature, _ = net(data1.cuda(non_blocking=True), data2.cuda(non_blocking=True), return_embedding=True)
            feature_bank.append(feature)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.cat(target_bank, dim=0).contiguous().to(feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data_tuple in test_bar:
            (data1, data2), target = data_tuple
            data1, data2, target = data1.cuda(non_blocking=True), data2.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, _ = net(data1, data2, return_embedding=True)

            total_num += data1.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data1.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data1.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data1.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

def parse_args():
    parser = argparse.ArgumentParser(description='Train BYOL network')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset: cifar10 or tiny_imagenet or stl10')
    parser.add_argument('--image_size', default=32, type=int, help='Size of image')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--proj_hidden_dim', default=512, type=int, help='Feature dim for latent vector')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--use_pretrained_enc', type=str2bool, default=False, help="use pretrained resnet 50 encoder")
    parser.add_argument('--use_default_enc', type=str2bool, default=True, help="use default resnet 50 encoder")
    parser.add_argument('--use_seed',
                        help='Should we set a seed for this particular run?',
                        type=str2bool,
                        default=False)
    parser.add_argument('--seed',
                        help='seed to fix in torch',
                        type=int,
                        default=0)
    parser.add_argument('--use_wandb',
                        help='remote logs to wandb',
                        type=str2bool,
                        default=True)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    print(torch.__version__)
    if args.use_seed:
        torch.manual_seed(args.seed)
    dataset = args.dataset
    batch_size, epochs = args.batch_size, args.epochs
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k

    # data prepare
    if dataset == 'cifar10':

        train_data = torchvision.datasets.CIFAR10(root='data', train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = True), download=True)
        memory_data = torchvision.datasets.CIFAR10(root='data', train=True, \
                                                  transform=utils.CifarPairTransform(train_transform = False), download=True)
        test_data = torchvision.datasets.CIFAR10(root='data', train=False, \
                                                  transform=utils.CifarPairTransform(train_transform = False), download=True)
    else:
        raise ValueError(f" Unknown dataset {dataset}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                            drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    
    if args.use_default_enc:
        encoder = models.resnet50(pretrained=args.use_pretrained_enc)
        hidden_layer = 'avgpool'
    else:
        enc_layers = []
        for name, module in models.resnet50(pretrained=args.use_pretrained_enc).named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if dataset == 'cifar10':
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                    enc_layers.append(module)
            elif dataset == 'tiny_imagenet' or dataset == 'stl10':
                if not isinstance(module, nn.Linear):
                    enc_layers.append(module)
        # encoder
        encoder = nn.Sequential(*enc_layers)
        hidden_layer = -1

    model = BYOL(net=encoder, 
                    image_size=args.image_size, 
                    hidden_layer=hidden_layer, 
                    projection_size=args.feature_dim, 
                    projection_hidden_size=args.proj_hidden_dim,
                    use_momentum=True).cuda()
    
    c = len(memory_data.classes)

    # if dataset == 'cifar10':
    #     flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(), torch.randn(1, 3, 32, 32).cuda()))

    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}'.format(args.feature_dim, args.proj_hidden_dim, batch_size, dataset)

    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        if epoch % 5 == 0:
            results['train_loss'].append(train_loss)
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(5, epoch + 1, 5))
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
            if test_acc_1 > best_acc:
                best_acc = test_acc_1
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        if epoch % 50 == 0:
            torch.save(model.state_dict(), 'results/{}_model_{}.pth'.format(save_name_pre, epoch))
