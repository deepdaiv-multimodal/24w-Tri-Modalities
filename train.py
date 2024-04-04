import os
import argparse
import collections

from dataset.msrvtt_dataloader import MSRVTT_DataLoader
from model.fusion_model_hk import EverythingAtOnceModel
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import DataLoader
from model.utils.cluster_loss import cluster_contrast

from torch.optim.lr_scheduler import StepLR

import time
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from functools import partial
import math
import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from fast_pytorch_kmeans import KMeans


def calculate_f1_score(predictions, labels):
    f1 = f1_score(labels.cpu(), predictions.cpu(), average='weighted')
    return f1


def TrainOneBatch(model, opt, data, loss_fun, apex=False, use_cls_token=False, centroid=None):
    video = data['video'].to(device)
    audio = data['audio'].to(device)
    text = data['text'].to(device)
    nframes = data['nframes'].to(device)
    category = data['category'].to(device)

    video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])
    nframes = nframes.view(-1)

    bs = video.size(0)
    # print('video:', video.shape, 'audio:', audio.shape, 'text:', text.shape)

    opt.zero_grad()
    
    #loss 
    if use_cls_token:
        v, a, t = model(video, audio, nframes, text, category)
        loss_v = F.cross_entropy(v, category)
        loss_a = F.cross_entropyloss_fun(a, category)
        loss_t = F.cross_entropyloss_fun(t, category)
        loss = loss_v + loss_a + loss_t
    else:
        # pred = model(video, audio, nframes, text, category)
        # loss = loss_fun(pred, category)
        va, at, tv, v, a, t  = model(video, audio, nframes, text, category) 
        # v = v.mean(dim=1)   
        # a = a.mean(dim=1) 
        # t = t.mean(dim=1) 
        fushed = (v + a + t) / 3

        kmeans = KMeans(bs, mode='cosine', verbose=1)
        labels = kmeans.fit_predict(fushed)
        centroid = kmeans.centroids
        # print('label:',labels,'centroid', centroid, 'fushed:', fushed)

        loss_v = F.cross_entropy(va, category)
        loss_a = F.cross_entropy(at, category)
        loss_t = F.cross_entropy(tv, category)
        loss_correlation = loss_v + loss_a + loss_t

        # loss_cluster = cluster_contrast(fushed, centroid, labels[-bs:], bs)

        # S = torch.matmul(fushed, centroid.t())
        # target = torch.zeros(bs, centroid.shape[0]).to(S.device)
        # target[range(target.shape[0]), labels] = 1
        # S = S - target * (0.001)
        # loss_cluster = F.nll_loss(F.log_softmax(S, dim=1), labels)

        loss_cluster_v = cluster_contrast(v, centroid, labels[-bs:],bs)
        loss_cluster_a = cluster_contrast(a, centroid, labels[-bs:],bs)
        loss_cluster_t = cluster_contrast(t, centroid, labels[-bs:],bs)
        loss_cluster = loss_cluster_v + loss_cluster_a + loss_cluster_t

        loss = loss_correlation + loss_cluster
        print('loss_correlation:', loss_correlation.item(), 'loss_cluster:', loss_cluster.item(), 'total loss:', loss.item())

    # loss_cluster.backward(retain_graph=True)
    # loss_correlation.backward()
    loss.backward()
    opt.step()
    return loss_cluster.item(), loss_correlation.item(),centroid

def get_soft_voting(va, at, tv):
    # Soft voting by averaging the probabilities
    soft_vote = (va + at + tv) / 3
    _, soft_vote_preds = torch.max(soft_vote, 1)
    return soft_vote_preds

def get_hard_voting(va_preds, at_preds, tv_preds):
    # Hard voting by selecting the most frequent prediction
    combined_preds = torch.stack((va_preds, at_preds, tv_preds), dim=1)
    hard_vote, _ = torch.mode(combined_preds, dim=1)
    return hard_vote

def get_predictions(va, at, tv):
    #va = torch.softmax(va, dim=1)
    #at = torch.softmax(at, dim=1)
    #tv = torch.softmax(tv, dim=1)

    _, va_preds = torch.max(va, 1)
    _, at_preds = torch.max(at, 1)
    _, tv_preds = torch.max(tv, 1)

    return va_preds, at_preds, tv_preds

def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

def EvalUseClsToken(val_batch, net):
    video = val_batch['video'].to(device)
    audio = val_batch['audio'].to(device)
    text = val_batch['text'].to(device)
    nframes = val_batch['nframes'].to(device)
    category = val_batch['category'].to(device)

    video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])

    va, at, tv = net(video, audio, nframes, text, category)
    va_preds, at_preds, tv_preds = get_predictions(va, at, tv)

    # Soft voting
    soft_vote_preds = get_soft_voting(va, at, tv)
    soft_vote_correct = (soft_vote_preds == category).sum().item()

    # Hard voting
    hard_vote_preds = get_hard_voting(va_preds, at_preds, tv_preds)
    hard_vote_correct = (hard_vote_preds == category).sum().item()

    # F1 Score
    f1 = calculate_f1_score(soft_vote_preds, category)  # 소프트 보팅의 예측 결과로 f1 score 계산
    

    # Calculate accuracy for each modality
    video_correct = (va_preds == category).sum().item()
    audio_correct = (at_preds == category).sum().item()
    text_correct = (tv_preds == category).sum().item()

    return video_correct, audio_correct, text_correct, soft_vote_correct, hard_vote_correct,f1

def EvalEmbedvec(val_batch, net):
    video = val_batch['video'].to(device)
    audio = val_batch['audio'].to(device)
    text = val_batch['text'].to(device)
    nframes = val_batch['nframes'].to(device)
    category = val_batch['category'].to(device)

    video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])

    pred = net(video, audio, nframes, text, category)
    _, pred = torch.max(pred.data, 1)
    correct = (pred == category).sum().item()
    return correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_path', default='C:/Users/heeryung/code/24w_deep_daiv/data/GoogleNews-vectors-negative300.bin', type=str)
    parser.add_argument('--data_path', default='C:/Users/heeryung/code/24w_deep_daiv/data/msrvtt_category_train.pkl', type=str)
    parser.add_argument('--val_data_path', default='C:/Users/heeryung/code/24w_deep_daiv/msrvtt_category_test.pkl', type=str)
    parser.add_argument('--save_path', default='C:/Users/heeryung/code/24w_deep_daiv/ckpt', type=str)
    parser.add_argument('--exp', default='trial1', type=str)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--use_softmax', default=True, type=bool)
    parser.add_argument('--use_cls_token', default=False, type=bool)
    parser.add_argument('--token_projection', default='projection_net', type=str)
    parser.add_argument('--num_classes', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--ckpt_path', default='', type=str)
    parser.add_argument('--resume', default=False, type=bool)
    #parser.add_argument('--device', default="3", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    # setup data_loader instances
    we=None
    we=KeyedVectors.load_word2vec_format(args.we_path, binary=True)

    save_path = args.save_path + '/' + args.exp
    os.makedirs(save_path, exist_ok=True)

    dataset = MSRVTT_DataLoader(data_path=args.data_path, we=we)
    val_dataset = MSRVTT_DataLoader(data_path=args.val_data_path,we=we)
    
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)#, num_workers=4)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)#, num_workers=4)
    print('# of train data:', len(data_loader), '# of valid data:', len(val_data_loader))

    loss = torch.nn.CrossEntropyLoss()
    net = EverythingAtOnceModel(args).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr =0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 매 10 에폭마다 학습률을 0.1배로 감소
    epoch = args.epoch

    total_video_correct = 0
    total_audio_correct = 0
    total_text_correct = 0
    total_hard_vote_correct = 0
    total_soft_vote_correct = 0
    total_f1=0
    total_correct = 0
    total_num = 0

    epochs_list = []
    accuracy_list = []
    hard_accuracy_list = []
    soft_accuracy_list = []
    f1_list = []

    centroid = None

    for epoch in range(0,epoch+1):
        net.train()
        running_loss = 0.0

        print('Epoch:', epoch)
        for i_batch, sample_batch in enumerate(data_loader):
            clustering_loss, correlation_loss, centroid = TrainOneBatch(net, optimizer, sample_batch, loss, use_cls_token=args.use_cls_token, centroid=centroid)
            batch_loss = clustering_loss + correlation_loss
            running_loss += batch_loss

        print('Epoch: {} / Total loss: {}'.format(epoch, running_loss / len(data_loader)))
        scheduler.step()

        if epoch % 10 == 0:
            # Save checkpoint
            checkpoint = {'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(save_path, 'epoch{}.pth'.format(epoch)))

            # validation 
            net.eval()
            with torch.no_grad():
                for val_batch in val_data_loader:
                    category = val_batch['category'].to(device)

                    if args.use_cls_token:
                        video_correct, audio_correct, text_correct, soft_vote_correct, hard_vote_correct = EvalUseClsToken(val_batch, net)

                        total_soft_vote_correct += soft_vote_correct
                        total_hard_vote_correct += hard_vote_correct
                        total_video_correct += video_correct
                        total_audio_correct += audio_correct
                        total_text_correct += text_correct
                    
                    else:
                        # correct = EvalUseEmbedvec(val_batch, net)
                        # total_correct += correct
                        video_correct, audio_correct, text_correct, soft_vote_correct, hard_vote_correct,f1score = EvalUseClsToken(
                            val_batch, net)
                        total_soft_vote_correct += soft_vote_correct
                        total_hard_vote_correct += hard_vote_correct
                        total_f1 += f1score
                        total_video_correct += video_correct
                        total_audio_correct += audio_correct
                        total_text_correct += text_correct

                    total_num += category.size(0)
                
                # Calculate final accuracies
                if args.use_cls_token:
                    print("Video accuracy:", total_video_correct / total_num)
                    print("Audio accuracy:", total_audio_correct / total_num)
                    print("Text accuracy:", total_text_correct / total_num)
                    print("Hard voting accuracy:", total_hard_vote_correct / total_num)
                    print("Soft voting accuracy:", total_soft_vote_correct / total_num)
                
                else: 
                  
                    hard_vote_accuracy = total_hard_vote_correct / total_num
                    soft_vote_accuracy = total_soft_vote_correct / total_num
                    f1_accuracy = total_f1 / total_num

                    print("Video accuracy:", total_video_correct / total_num)
                    print("Audio accuracy:", total_audio_correct / total_num)
                    print("Text accuracy:", total_text_correct / total_num)

                    epochs_list.append(epoch)
                    hard_accuracy_list.append(hard_vote_accuracy)
                    soft_accuracy_list.append(soft_vote_accuracy)
                    f1_list.append(f1_accuracy)

                    plt.figure(figsize=(10, 6))
                    plt.plot(epochs_list, soft_accuracy_list, marker='o', linestyle='-', color='r', label='Soft Voting')
                    plt.title('Accuracy over Epochs')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.grid(True)
                    plt.savefig(save_path + '/accuracy_epoch{}.png'.format(epoch))
                    plt.close()