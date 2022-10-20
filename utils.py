

import numpy as np
import torch
import copy

device = torch.device('cuda')


def get_acc(model, data_loader):
  running_acc = 0.0
  instances = 0
  for i, data in enumerate(data_loader):
    inputs, labels = data
    inputs = inputs.to(device, dtype=torch.double)
    labels = labels.to(device)
    instances += len(labels)
    outputs = model(inputs)
    _, p = torch.max(outputs, 1)
    running_acc += (p == labels).sum()
  return running_acc / instances


def get_ob_w_feat_val(feature_index, value, df, num):
  obs = []
  col = df.iloc[:, feature_index].to_numpy()
  w = np.where(col == value)[0]
  return np.random.choice(w, size=num)


def get_all_combinations(arr, list_output, index_arr, index, current_vals):
  if (index >= len(index_arr)):
    deep_list = list(current_vals)
    list_output.append(deep_list)
    return
  current_index = index_arr[index]
  # print(index, current_index)
  for i in range(0, len(arr[index])):
    current_vals.append(arr[index][i])
    get_all_combinations(arr, list_output, index_arr, index + 1, current_vals)
    current_vals.pop()
  return


def prob_w_values(values, indices, df):
  s = df.shape[0]
  dist = []

  print('vals: ', values)

  for i in range(0, len(values)):
    v = values[i]
    p = []
    for j in range(0, len(v)):
      feat_val = v[j]
      index = indices[j]
      p.append(np.where(df.iloc[:, index].to_numpy() == feat_val))

    # print(p)
    if (len(p) > 1):
      intersect = np.intersect1d(p[0], p[1])
    else:
      intersect = p[0][0]
    # print(len(intersect))
    for j in range(2, len(p)):
      intersect = np.intersect1d(intersect, p[j])
    dist.append(len(intersect) / s)
  # print('dist',dist)
  return dist


def multi_feature_split(indices, alpha, df, num_agents):
  agent_data_sets = {}
  filtered_df = copy.deepcopy(df)
  # print(filtered_df.head())
  num_observations = round(filtered_df.shape[0] / num_agents)
  unique_vals = list(map(lambda x: np.unique(filtered_df.iloc[:, x].to_numpy()), indices))
  l = []
  get_all_combinations(unique_vals, l, indices, 0, [])
  dist = np.array(prob_w_values(l, indices, filtered_df))
  ran = np.arange(0, len(l))
  r = list(map(lambda x: (l[x], dist[x]), ran))
  indices_w_nonzero = np.where(dist > 0.0)
  # print(indices_w_nonzero)
  vals = np.array(r)[indices_w_nonzero]
  a = np.array(list(map(lambda x: x[1], vals)))
  for i in range(0, num_agents):
    dir_dist = np.random.dirichlet(alpha=np.multiply(a, alpha))
    print('dist: ', dir_dist)
    vals = list(map(lambda x: (vals[x][0], vals[x][1], dir_dist[x]), np.arange(0, len(vals))))
    print('vals: ',vals)
    multi = get_sample_multivariate(num_observations, vals)
    samples = get_n_random_samples(filtered_df, indices, vals, multi, num_observations)
    agent_data_sets[i] = samples
  # multi = get_sample_multivariate(10,vals)
  # get_random_sample_from_dataset(df,vals[multi[0]][0],indices)
  return agent_data_sets


# agent_sets = single_feature_split(1, 1, 10, adults_df)


def get_n_random_samples(df, feature_indices, vals, multi, n):
  samples = []
  freq = {}
  for i in range(0, n):
    # print(multi[i])
    if multi[i] not in freq:
      freq[multi[i]] = 1
    else:
      freq[multi[i]] += 1

  for key in freq:
    # print(key)
    s = get_random_sample_from_dataset(df, vals[key][0], feature_indices, freq[key])
    samples = samples + s
  # for i in range(0,n):
  #   print(multi[i])
  #   #print(vals[multi[i]][0])
  #   s = get_random_sample_from_dataset(df,vals[multi[i]][0],feature_indices)
  #   samples.append(s)
  return samples


def get_random_sample_from_dataset(df, feature_vals, feature_indices, times):
  p = []
  i = 0
  for index in feature_indices:
    p.append(df.iloc[:, index] == feature_vals[i])
    i += 1
  t = df.iloc[:, feature_indices[0]] != None

  for j in range(0, len(p)):
    t = t & p[j]

  return np.random.choice(df.loc[t].index, size=times).tolist()


def get_sample_multivariate(n, dist):
  dir_dist = np.array(list(map(lambda x: x[2], dist)))
  vals = np.arange(0, len(dist))
  choices = []
  for i in range(0, n):
    choices.append(np.random.choice(vals, p=dir_dist))
  # print(choices)
  return choices


def get_sample_val(vals, dist):
  np.random.random_sample
  return np.random.choice(vals, p=dist)


def sample_n_times(n, dist, vals):
  freq = {}
  for i in range(0, n):
    val = get_sample_val(vals, dist)
    if (val not in freq):
      freq[val] = 1
    else:
      freq[val] += 1
  return freq


def single_feature_split(feature_index, alpha, df, num_agents):
  num_observations = round(df.shape[0] / num_agents)
  fs = []
  for i in range(0, num_agents):
    vals, dist = dirichlet_over_col(feature_index, df, alpha)
    f = sample_n_times(num_observations, dist, vals)
    # print(f)
    fs.append(f)
  agent_data_sets = {}
  for i in range(0, num_agents):
    freq = fs[i]
    s = []
    for key in freq:
      s.extend(get_ob_w_feat_val(feature_index, key, df, freq[key]))
    agent_data_sets[i] = s
  return agent_data_sets


def dirichlet_over_col(feature_index, df, alpha):
  col = df.iloc[:, feature_index].to_numpy()
  vals = np.unique(col)
  prior_probability_dist = []
  for i in range(0, len(vals)):
    k = len(np.where(col == vals[i])[0])
    prior_probability_dist.append(k / len(col))
  # print(alpha)
  dist = np.random.dirichlet(np.multiply(prior_probability_dist, alpha))
  print('Dirichlet Dist: ', dist)
  return (vals, dist)


def get_sample_val(vals, dist):
  np.random.random_sample
  return np.random.choice(vals, p=dist)


