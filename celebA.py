import pandas as pd
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch
import copy
import math
import torchvision
import random

import torchvision.transforms as transforms
import utils
from agents import Agent
import aggregate

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
            nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),
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
        feature = feature.view(-1, 16 * 13 * 13)
        output = self.fc(feature)
        return output

def celeba_torch_to_df(celeb_data):
    attr_tensors = []
    for i in range(0, len(celeb_data)):
        attr_tensors.append(celeb_data[i][1].numpy())
    #attr_tensors = celeb_data[:,1].numpy()
    return pd.DataFrame(attr_tensors)
def get_dataset_for_agent(celeb_data, indices,target_index,is_poisoned=False):

    d = []
    if not is_poisoned:
        for index in indices:
            pic = celeb_data[index][0]
            attr= celeb_data[index][1][target_index]
            d.append((pic,attr))
    else:
        for index in indices:
            pic = celeb_data[index][0]
            attr= celeb_data[index][1]
            d.append((pic,torch.tensor(attr)))

    return d

def agents(num_agents, dataset,alpha,feature_split,poisoned_set,m):
    df = celeba_torch_to_df(dataset)
    target_index = 31
    agent_data_indices = utils.multi_feature_split(feature_split,alpha,df,num_agents)
    num_bad_apples = math.floor(m*num_agents)
    a = {}
    for key in agent_data_indices:
        if(num_bad_apples > 0):
            dl = DataLoader(get_dataset_for_agent(poisoned_set,agent_data_indices[key],target_index,is_poisoned=True), batch_size=32)
            num_bad_apples-=1
        else:
            dl = DataLoader(get_dataset_for_agent(dataset,agent_data_indices[key],target_index), batch_size=32)

        a[key] = Agent(dl)
    return a

def backdoor_img(x):
    x = np.array(x.squeeze())
    for i in range(3):
        for j in range(21,26):
            for z in range(21,26):
                x[i,j,z]=(6*0.5)+0.5
    return x

def poison_dataset(dataset, base_class,poison_frac):
    all_idxs = np.arange(0,len(dataset)).tolist()

    poison_idxs = random.sample(all_idxs,math.floor(poison_frac*len(all_idxs)))
    for idx in poison_idxs:
        clean_img = dataset[idx][0]
        bd_img = backdoor_img(clean_img)
        label = base_class
        dataset[idx] = (torch.tensor(bd_img),label)

    print(f'poisoned dataset with {math.floor(poison_frac*len(all_idxs))} backdoor instances :)')
    return poison_idxs


def get_backdoor_acc(backdoor_indices,dataset, model):
    correct = 0.0
    inps = []
    lbls = []
    for index in backdoor_indices:
        inp = dataset[index][0]
        lbl = dataset[index][1]
        inps.append(inp)
        lbls.append(torch.tensor(lbl))
    ten_inps = torch.Tensor(len(inps))
    inps = torch.stack(inps)
    lbls = torch.stack(lbls)

    output = model(inps)
    _, p = torch.max(output, 1)


    return (p==lbls).sum()/len(backdoor_indices)






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
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


celeb_data = torchvision.datasets.CelebA('celeb_root',target_type='attr', download=False, transform=train_transforms, split='train')
celeb_data_test = torchvision.datasets.CelebA('celeb_root',target_type='attr',download=False, transform=train_transforms,split='test')


print(f'Rows of celeb loaded: {len(celeb_data)}')
print(f'Rows of test celeb loaded: {len(celeb_data_test)}')


test_loader = DataLoader(get_dataset_for_agent(celeb_data_test,np.arange(0, len(celeb_data_test)),31), batch_size=32)
poison_set = get_dataset_for_agent(celeb_data,np.arange(0,len(celeb_data)),31)
poison_dataset(poison_set,1,.5)
print('poisoning the test set...')
poison_eval_set = get_dataset_for_agent(celeb_data_test, np.arange(0,len(celeb_data_test)),31)
eval_poison_idxs = poison_dataset(poison_eval_set,1,.5)
print('poisoning complete starting report...')
#print(get_backdoor_acc(eval_poison_idxs,poison_eval_set,CELEB_NET()))

def full_report():
    alphas = [1e-1,1,1e2,1e3,1e4,1e5]
    bad_apple_fractions = [.1,.3,.5,.7]
    agg_methods = ['comed','sign','avg']
    for agg in agg_methods:
        for bad_apple_frac in bad_apple_fractions:
            fig, axs = plt.subplots(3,2)
            fig_poison, axs_poison = plt.subplots(3,2)
            row = 0
            col = 0
            m = bad_apple_frac
            n=0.1
            for alpha in alphas:
                agts = agents(100, celeb_data,alpha,[21,36],poison_set,m)
                model = CELEB_NET()
                criterion = nn.CrossEntropyLoss()
                print('Initiating Training...')
                accs,poison_acc = global_train(model, criterion, agts, n, test_loader, celeb_data,rounds=250,agg=agg,poison_eval_set=poison_eval_set,poison_indices=eval_poison_idxs)
                axs[row,col].plot(accs.keys(),accs.values())
                axs[row,col].set_title(f'{agg.upper()}: Accuracy VS Rounds N=100 n={n*100}% m={m*100}% Alpha={alpha}')
                axs[row,col].set(xlabel='Round',ylabel='Accuracy')
                axs[row,col].set_ylim([0,1])
                axs_poison[row,col].plot(poison_acc.keys(),poison_acc.values())
                axs_poison[row,col].set_title(f'{agg.upper()}: Backdoor Accuracy VS Rounds N=100 n={n*100}% m={m*100}% Alpha={alpha} p={50}%')
                axs_poison[row,col].set(xlabel='Round', ylabel='Backdoor Accuracy')
                axs_poison[row,col].set_ylim([0,1])
                col+=1
                if(col>=2):
                    col=0
                    row+=1
        plt.show()

def report():
    alphas = [1e-1,1,1e2,1e3,1e4,1e5]
    fig, axs = plt.subplots(3,2)
    fig_poison, axs_poison = plt.subplots(3,2)
    row = 0
    col = 0
    m = 0.1
    n=0.1
    for alpha in alphas:
        agts = agents(100, celeb_data,alpha,[21,36],poison_set,m)
        model = CELEB_NET()
        criterion = nn.CrossEntropyLoss()
        print('Initiating Training...')
        accs,poison_acc = global_train(model, criterion, agts, n, test_loader, celeb_data,rounds=250,agg='avg',poison_eval_set=poison_eval_set,poison_indices=eval_poison_idxs)
        axs[row,col].plot(accs.keys(),accs.values())
        axs[row,col].set_title(f'FED AVG: Accuracy VS Rounds N=100 n={n*100}% m={m*100}% Alpha={alpha}')
        axs[row,col].set(xlabel='Round',ylabel='Accuracy')
        axs[row,col].set_ylim([0,1])
        axs_poison[row,col].plot(poison_acc.keys(),poison_acc.values())
        axs_poison[row,col].set_title(f'FED AVG: Backdoor Accuracy VS Rounds N=100 n={n*100}% m={m*100}% Alpha={alpha} p={50}%')
        axs_poison[row,col].set(xlabel='Round', ylabel='Backdoor Accuracy')
        axs_poison[row,col].set_ylim([0,1])
        col+=1
        if(col>=2):
            col=0
            row+=1
    plt.show()
        # plt.plot(accs.keys(),accs.values())
        # plt.title('FED AVG: Accuracy VS Rounds  N=100 n=10% m=0.0')
        # plt.xlabel('Round')
        # plt.ylabel('Accuracy')
        # plt.show()
full_report()

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








