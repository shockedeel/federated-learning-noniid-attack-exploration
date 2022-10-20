import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch
import copy
import math
import torchvision
import random
from torchsummary import summary

import torchvision.transforms as transforms
import utils
from agents import Agent
import aggregate

matplotlib.use('TkAgg')
print(torch.cuda.is_available())
device = torch.device('cuda')
NUM_CLASSES=2
GRAYSCALE=False
class CELEB_NET(nn.Module):
    def __init__(self):
        super(CELEB_NET, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16,26,5),
            nn.BatchNorm2d(26),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(832//2, 120),
            nn.BatchNorm1d(120),
            nn.LeakyReLU(),

            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.LeakyReLU(),

            nn.Linear(84, 2),
            nn.Sigmoid()

        )

    def forward(self, input):
        feature = self.conv(input)
        feature = feature.view(-1,832//2)
        output = self.fc(feature)
        return output

def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = torchvision.models.resnet18(pretrained=False)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, 2, bias=True)

    return model
resnet = resnet18(NUM_CLASSES)
summary(resnet,(3,64,64))

def celeba_torch_to_df(celeb_data):
    attr_tensors = []
    for i in range(0, len(celeb_data)):
        attr_tensors.append(celeb_data[i][1].numpy())
    #attr_tensors = celeb_data[:,1].numpy()
    df =  pd.DataFrame(attr_tensors)
    print('index 32: ', df.iloc[:, 32].value_counts())
    print('index 31: ', df.iloc[:, 31].value_counts())
    return df
def get_dataset_for_agent(celeb_data, indices,target_index,is_poisoned=False):
    if(is_poisoned):

        indices = np.random.choice(np.arange(0,len(celeb_data)),size=len(indices),replace=False)

    d = []
    if not is_poisoned:
        for index in indices:
            pic = celeb_data[index][0]
            attr= celeb_data[index][1][target_index]
            d.append((pic,torch.tensor(attr)))
    else:
        for index in indices:
            pic = celeb_data[index][0]
            attr= celeb_data[index][1]
            d.append((pic,torch.tensor(attr)))

    return d

def get_label_split_agents(num_agents, dataset, label_index,het=False):
    a = {}
    df = celeba_torch_to_df(dataset)
    per_client = math.floor(df.shape[0]/10)
    valid_sample_indices = np.arange(0, df.shape[0])
    index = df.index
    for i in range(num_agents):

        label_val = np.random.choice([0,1])
        condition = df.iloc[:,label_index]==label_val
        valid_filtered = index[condition]
        if(het):
            l = np.random.choice(valid_filtered,size=per_client)
        else:
            l = np.random.choice(valid_sample_indices,size=per_client)
        ds = get_dataset_for_agent(dataset,l,label_index)
        dl = DataLoader(ds,batch_size=128)
        a[i] = Agent(dl,is_malicious=False)
    return a
def agents(num_agents, dataset,alpha,feature_split,poisoned_set,m):
    df = celeba_torch_to_df(dataset)

    target_index = 31
    print('multifeature: ',feature_split)
    agent_data_indices = utils.multi_feature_split(feature_split,alpha,df,num_agents)

    num_bad_apples = math.floor(m*num_agents)
    a = {}
    for key in agent_data_indices:
        was_bad = False
        if(num_bad_apples > 0):
            dl = DataLoader(get_dataset_for_agent(poisoned_set,agent_data_indices[key],target_index,is_poisoned=True), batch_size=128)
            num_bad_apples-=1
            was_bad = True
        else:
            testing = []
            for i in range(5):
                testing.append(agent_data_indices[key][i])
            print(f'testing: {testing}')
            ds = get_dataset_for_agent(dataset, agent_data_indices[key], target_index)

            dl = DataLoader(ds, batch_size=128)
        if(was_bad):
            a[key] = Agent(dl,is_malicious=True)
        else:
            a[key] = Agent(dl, is_malicious=False)
            print('label dist: ',a[key].get_label_dist())
    return a

def backdoor_img(x, type='default'):
    x = np.array(x.squeeze())
    if(type == 'default'):
        for i in range(3):
            for j in range(21,26):
                for z in range(21,26):
                    x[i,j,z]=255
    if(type == 'bigger_square'):
        for i in range(3):
            for j in range(21,36):
                for z in range(21, 36):
                    x[i,j,z] = 255
    if(type == 'plus'):
        start_idx = 5
        size = 10
        for channel in range(3):
            for i in range(start_idx, start_idx+size):
                x[channel, i, start_idx] = 255
        for channel in range(3):
            for i in range(start_idx-size//2, start_idx+size//2+1):
                x[channel, start_idx+size//2, i]=255
    return x

def poison_dataset(dataset, base_class,target_class,poison_frac):
    all_idxs = []
    for i in range(len(dataset)):
        if(dataset[i][1]==base_class):
            all_idxs.append(i)

    poison_idxs = random.sample(all_idxs,math.floor(poison_frac*len(all_idxs)))
    for idx in poison_idxs:
        clean_img = dataset[idx][0]
        bd_img = backdoor_img(clean_img,type='plus')
        label = target_class
        dataset[idx] = (torch.tensor(bd_img),label)

    print(f'poisoned dataset with {math.floor(poison_frac*len(all_idxs))} backdoor instances :)')
    return poison_idxs



def get_backdoor_acc(backdoor_indices,dataset, model):

    d = []
    for index in backdoor_indices:
        inp = dataset[index][0]
        lbl = dataset[index][1]
        d.append((inp,torch.tensor(lbl)))

    dl = DataLoader(d,batch_size=256)
    num_instances = 0
    correct = 0
    for _, (inputs, labels) in enumerate(dl):
        num_instances += len(labels)
        inputs = inputs.to(device, dtype=torch.double)
        labels = labels.to(device)
        outputs = model(inputs)
        _,p=torch.max(outputs, 1)
        correct += (p == labels).sum()






    return correct/num_instances






    return correct/len(backdoor_indices)

def global_train(global_model,criterion, agents,agent_frac,test_loader,X,rounds=10,agg='avg',poison_eval_set=None, poison_indices=None):
  n_agents = len(agents)
  data_size = len(X)/n_agents
  accs = {}
  poison_acc = {}
  print(f'AGENT_FRAC: {agent_frac}, N_AGENTS: {n_agents}')
  for round in range(0, rounds):
    print(f'ROUND: {round}')
    round_global_params = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
    agent_updates = {}
    for agent in np.random.choice(n_agents,math.floor(agent_frac*n_agents),replace=False):
      agent_updates[agent] = agents[agent].train_local(global_model,criterion)
      torch.nn.utils.vector_to_parameters(copy.deepcopy(round_global_params),global_model.parameters())
    aggregate.aggregate(global_model,agent_updates,agg,data_size)
    accs[round+1] = utils.get_acc(global_model, test_loader)
    if(poison_eval_set != None):
        poison_acc[round+1] = get_backdoor_acc(poison_indices,poison_eval_set,global_model)
        print(f"POISON ACCURACY: {poison_acc[round+1]}")
    print(f"ROUND ACCURACY: {accs[round+1]}")


  if(poison_eval_set != None):
      return (accs, poison_acc)
  return accs





image_size = 64
train_transforms =  transforms.Compose([transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()])


celeb_data = torchvision.datasets.CelebA('celeb_root',target_type='attr', download=True, transform=train_transforms, split='train')
celeb_data_test = torchvision.datasets.CelebA('celeb_root',target_type='attr',download=False, transform=train_transforms,split='test')

#cnn = CELEB_NET().to(device, dtype=torch.double)
#test_agents = agents(100, celeb_data,1,[21,36],None,0.0)
#criterion = torch.nn.CrossEntropyLoss()

print(f'Rows of celeb loaded: {len(celeb_data)}')
print(f'Rows of test celeb loaded: {len(celeb_data_test)}')


test_loader = DataLoader(get_dataset_for_agent(celeb_data_test,np.arange(0, len(celeb_data_test)),31), batch_size=32)
#global_train(cnn,criterion,test_agents,.1,test_loader,celeb_data,rounds=10,agg='avg')
#exit(1)
#poison_set = get_dataset_for_agent(celeb_data,np.arange(0,len(celeb_data)),31)
#poison_dataset(poison_set,0,1,1.0)
print('poisoning the test set...')
#poison_eval_set = get_dataset_for_agent(celeb_data_test, np.arange(0,len(celeb_data_test)),31)
#eval_poison_idxs = poison_dataset(poison_eval_set,0,1,1.0)

print('poisoning complete starting report...')
#print(get_backdoor_acc(eval_poison_idxs,poison_eval_set,CELEB_NET()))

def full_report(indices):
    alphas = [1e4]
    bad_apple_fractions = [0.0]
    agg_methods = ['avg']
    for agg in agg_methods:
        for bad_apple_frac in bad_apple_fractions:
            fig, axs = plt.subplots(2,1)
            fig_poison, axs_poison = plt.subplots(2,1)
            row = 0
            m = bad_apple_frac
            n=1.0
            for alpha in alphas:
                #agts = agents(10, celeb_data,alpha,indices,None,m)
                agts = get_label_split_agents(10,celeb_data,31,het=True)
                model = resnet18(2).to(device, dtype=torch.double)
                criterion = nn.CrossEntropyLoss()
                print('Initiating Training...')
                accs,poison_acc = global_train(model, criterion, agts, n, test_loader, celeb_data,rounds=50,agg=agg)
                axs[row].plot(accs.keys(),accs.values())
                axs[row].set_title(f'{agg.upper()}: Accuracy VS Rounds Indices={indices} Alpha={alpha}')
                axs[row].set(xlabel='Round',ylabel='Accuracy')
                axs[row].set_ylim([0,1])
                axs_poison[row].plot(poison_acc.keys(),poison_acc.values())
                axs_poison[row].set_title(f'{agg.upper()}: Backdoor Accuracy VS Rounds N=100 n={n*100}% m={m*100}% Alpha={alpha} p={50}%')
                axs_poison[row].set(xlabel='Round', ylabel='Backdoor Accuracy')
                axs_poison[row].set_ylim([0,1])
                row+=1

        plt.show()

# def report():
#     alphas = [1e-1,1,1e2,1e3,1e4,1e5]
#     fig, axs = plt.subplots(3,2)
#     fig_poison, axs_poison = plt.subplots(3,2)
#     row = 0
#     col = 0
#     m = 0.1
#     n=0.1
#     for alpha in alphas:
#         agts = agents(100, celeb_data,alpha,[21,36],poison_set,m)
#         model = CELEB_NET()
#         criterion = nn.CrossEntropyLoss()
#         print('Initiating Training...')
#         accs,poison_acc = global_train(model, criterion, agts, n, test_loader, celeb_data,rounds=250,agg='avg',poison_eval_set=poison_eval_set,poison_indices=eval_poison_idxs)
#         axs[row,col].plot(accs.keys(),accs.values())
#         axs[row,col].set_title(f'FED AVG: Accuracy VS Rounds N=100 n={n*100}% m={m*100}% Alpha={alpha}')
#         axs[row,col].set(xlabel='Round',ylabel='Accuracy')
#         axs[row,col].set_ylim([0,1])
#         axs_poison[row,col].plot(poison_acc.keys(),poison_acc.values())
#         axs_poison[row,col].set_title(f'FED AVG: Backdoor Accuracy VS Rounds N=100 n={n*100}% m={m*100}% Alpha={alpha} p={50}%')
#         axs_poison[row,col].set(xlabel='Round', ylabel='Backdoor Accuracy')
#         axs_poison[row,col].set_ylim([0,1])
#         col+=1
#         if(col>=2):
#             col=0
#             row+=1
#     plt.show()
#         # plt.plot(accs.keys(),accs.values())
#         # plt.title('FED AVG: Accuracy VS Rounds  N=100 n=10% m=0.0')
#         # plt.xlabel('Round')
#         # plt.ylabel('Accuracy')
#         # plt.show()
full_report([5,31])
#full_report([21,36])#BALANCED INDICES
#LABEL INDEX

# for epoch in range(100):
#     epoch_loss = 0.0
#     for _, (inputs, labels) in enumerate(t_l):
#         optimizer.zero_grad()
#         outputs=model(inputs)
#         loss = criterion(outputs, labels)
#         #_, p = torch.max(outputs, 1)
#         epoch_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     print(f'EPOCH LOSS {epoch_loss}')








