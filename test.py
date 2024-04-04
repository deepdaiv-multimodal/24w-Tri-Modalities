import torch
import torch.nn as nn
from dataset.msrvtt_dataloader import MSRVTT_DataLoader
from model.fusion_model_hk import EverythingAtOnceModel
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import DataLoader

import argparse

<<<<<<< HEAD
parser = argparse.ArgumentParser()
parser.add_argument('--we_path', default='C:/Users/heeryung/code/24w_deep_daiv/GoogleNews-vectors-negative300.bin', type=str)
parser.add_argument('--data_path', default='C:/Users/heeryung/code/24w_deep_daiv/msrvtt_category_test.pkl', type=str)
parser.add_argument('--checkpoint_path', default='D:/download/epoch200.pth', type=str)
parser.add_argument('--token_projection', default='projection_net', type=str) 
parser.add_argument('--use_softmax', default=False, type=bool) 
parser.add_argument('--use_cls_token', default=False, type=bool) 
parser.add_argument('--num_classes', default=20, type=int) 
parser.add_argument('--batch_size', default=16, type=int) 
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint_path)

we = None 
we = KeyedVectors.load_word2vec_format(args.we_path, binary=True)

dataset = MSRVTT_DataLoader(data_path=args.data_path, we=we)
data_loader = DataLoader(dataset, batch_size=args.batch_size)

net = EverythingAtOnceModel(args).cuda()
# optimizer = torch.optim.Adam(net.parameters(), lr =0.001)

net.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net.eval()
=======
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_path', default='GoogleNews-vectors-negative300.bin', type=str)
    parser.add_argument('--data_path', default='msrvtt_category_test.pkl', type=str)
    parser.add_argument('--checkpoint_path', default='epoch210.pth', type=str)
    parser.add_argument('--token_projection', default='projection_net', type=str) 
    parser.add_argument('--use_softmax', default=False, type=bool) 
    parser.add_argument('--use_cls_token', default=False, type=bool) 
    parser.add_argument('--num_classes', default=20, type=int) 
    parser.add_argument('--batch_size', default=16, type=int) 
    args = parser.parse_args()
    return args
>>>>>>> b68c6485e1a92903be0f721cc1bd770ae30ce8e2

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
    _, va_preds = torch.max(va, 1)
    _, at_preds = torch.max(at, 1)
    _, tv_preds = torch.max(tv, 1)
    return va_preds, at_preds, tv_preds

def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

<<<<<<< HEAD
def get_predictions(va, at, tv):
    #va = torch.softmax(va, dim=1)
    #at = torch.softmax(at, dim=1)
    #tv = torch.softmax(tv, dim=1)

    _, va_preds = torch.max(va, 1)
    _, at_preds = torch.max(at, 1)
    _, tv_preds = torch.max(tv, 1)

    return va_preds, at_preds, tv_preds

total_samples = 0
total_accuracy = 0
total_video_correct = 0
total_audio_correct = 0
total_text_correct = 0
total_hard_vote_correct = 0
total_soft_vote_correct = 0

for data in data_loader:
    video = data['video'].cuda()
    audio = data['audio'].cuda()
    text = data['text'].cuda()
    nframes = data['nframes'].cuda()
    category = data['category'].cuda() # [batch_size,]
    video_id = data['video_id']
=======
if __name__ == '__main__':
    args = parse_args()

    checkpoint = torch.load(args.checkpoint_path)
>>>>>>> b68c6485e1a92903be0f721cc1bd770ae30ce8e2

    we = None 
    we = KeyedVectors.load_word2vec_format(args.we_path, binary=True)

<<<<<<< HEAD
    va, at, tv = net(video, audio, nframes, text, category) # [batch_size, 20]
    va_preds, at_preds, tv_preds = get_predictions(va, at, tv) # [batch_size,]
    # accuracy = torch.mean((pred_category == category).float()) # [batch_size,]
    print(va_preds, '/', at_preds, '/', tv_preds, '/', category, '/', video_id)
    print('-------------------------------------------------------------')

#     total_accuracy += accuracy

# # Calculate final accuracies
# accuracy = total_accuracy / len(data_loader)

# print("Accuracy:", accuracy)
=======
    dataset = MSRVTT_DataLoader(data_path=args.data_path, we=we)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    net = EverythingAtOnceModel(args).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr =0.001)

    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.eval()

    total_samples = 0
    total_accuracy = 0
    total_video_correct = 0
    total_audio_correct = 0
    total_text_correct = 0
    total_hard_vote_correct = 0
    total_soft_vote_correct = 0

    for data in data_loader:
        video = data['video'].cuda()
        audio = data['audio'].cuda()
        text = data['text'].cuda()
        nframes = data['nframes'].cuda()
        category = data['category'].cuda() # [batch_size,]

        video = video.view(-1, video.shape[-1])
        audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
        text = text.view(-1, text.shape[-2], text.shape[-1])

        pred = net(video, audio, nframes, text, category) # [batch_size, 20]
        pred_category = torch.argmax(pred, dim=1) # [batch_size,]
        accuracy = torch.mean((pred_category == category).float()) # [batch_size,]
        print(pred_category, '/', category)

        total_accuracy += accuracy

    # Calculate final accuracies
    accuracy = total_accuracy / len(data_loader)
    print("Accuracy:", accuracy)
>>>>>>> b68c6485e1a92903be0f721cc1bd770ae30ce8e2
