
import torch


def agg_avg(agent_updates_dict,agent_data_sizes):
  sm_update = 0.0

  for agent in agent_updates_dict:
    up = agent_updates_dict[agent]
    sm_update += agent_data_sizes*up
  return sm_update/(agent_data_sizes*len(agent_updates_dict))

def agg_comed(agent_updates_dict):
  agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
  concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
  return torch.median(concat_col_vectors, dim=1).values
def agg_sign(agent_updates_dict):
  agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
  sm_signs = torch.sign(sum(agent_updates_sign))
  return torch.sign(sm_signs)

def aggregate(global_model,agent_updates_dict, t, agent_data_sizes):
  update = 0.0
  lr = .1
  if(t == 'avg'):
    update = agg_avg(agent_updates_dict, agent_data_sizes)
  if(t =='comed'):
    update=agg_comed(agent_updates_dict)
  if(t == 'sign'):
    update=agg_sign(agent_updates_dict)

  cur_global_params = torch.nn.utils.parameters_to_vector(global_model.parameters())
  new_global_params =  (cur_global_params + lr*update).double()
  torch.nn.utils.vector_to_parameters(new_global_params, global_model.parameters())
  return
