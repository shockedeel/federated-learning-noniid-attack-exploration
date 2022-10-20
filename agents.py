
import torch
import torch.nn as nn
import math
import utils

class Agent():
  def __init__ (self, train_loader, is_malicious=False):

    self.device = torch.device('cuda')
    self.train_loader = train_loader
    self.epochs = 1
    self.malicious = is_malicious
  def get_label_dist(self):
      index = 31
      dist =[0,0]
      total = 0
      for _, (inputs,labels) in enumerate(self.train_loader):
          total+=len(labels)
          dist[0]=len(labels[labels==0])+dist[0]
          dist[1]=len(labels[labels==1])+dist[1]
      dist[0]/=total
      dist[1]/=total
      return dist


  def train_local(self,global_model,criterion):
      print('training local...')
      #print(global_model)
      #print(criterion)
      initial_global_model_params = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
      #print(initial_global_model_params)

      global_model.train()
      optimizer = torch.optim.SGD(global_model.parameters(), lr = 0.1)
      #print("Starting local train:")
      total_loss=0.0
      for i in range(0,self.epochs):

        running_loss = 0.0
        for _, (inputs, labels) in enumerate(self.train_loader):
          inputs = inputs.to(self.device, dtype=torch.double)
          labels = labels.to(self.device)
          optimizer.zero_grad()
          outputs = global_model(inputs)
          batch_loss = criterion(outputs,labels)
          running_loss+=batch_loss.item()
          total_loss+=running_loss
          batch_loss.backward()

          nn.utils.clip_grad_norm_(global_model.parameters(),10)
          optimizer.step()
        #print(f'Local Loss: {running_loss}')
        print(f'local acc: ', utils.get_acc(global_model, self.train_loader))
      with torch.no_grad():
          final_global_model_parameters = torch.nn.utils.parameters_to_vector(global_model.parameters()).double()

          update = (final_global_model_parameters - initial_global_model_params)
          if(self.malicious):#Need to boost the norm of malicious agents so we don't forget the backdoor
              update = torch.mul(update, 1)
          return update
