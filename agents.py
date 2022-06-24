
import torch
import torch.nn as nn

class Agent():
  def __init__ (self, train_loader):


    self.train_loader = train_loader
    self.epochs = 2

  def train_local(self,global_model,criterion):
      #print(global_model)
      #print(criterion)
      initial_global_model_params = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()
      #print(initial_global_model_params)

      global_model.train()
      optimizer = torch.optim.SGD(global_model.parameters(), lr = 0.1)
      #print("Starting local train:")
      for i in range(0,self.epochs):

        running_loss = 0.0
        for _, (inputs, labels) in enumerate(self.train_loader):
          optimizer.zero_grad()
          outputs = global_model(inputs)
          batch_loss = criterion(outputs,labels)
          running_loss+=batch_loss.item()
          batch_loss.backward()

          nn.utils.clip_grad_norm_(global_model.parameters(),10)
          optimizer.step()
        #print(f'Local Loss: {running_loss}')

      final_global_model_parameters = torch.nn.utils.parameters_to_vector(global_model.parameters()).double()

      return final_global_model_parameters - initial_global_model_params
