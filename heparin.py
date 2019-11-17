###########################
# Latent ODEs for Heparin Management
# Author: Scotty Fleming
# Adapted from Latent ODEs for Irregularly-Sampled Time Series
#     by Yulia Rubanova and Ricky Chen
# (in turn adapted from: github.com/rtqichen/time-series-datasets)
###########################

import torch
from datetime import datetime
from collections import defaultdict
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_header(line):
    params_in_file = line.split(',')
    params_in_file = [str.replace(s, '"', '').strip() for 
                      s in params_in_file]
    params_to_indx = {p: i for i, p in enumerate(params_in_file)}
    params_to_indx = {p: i for p, i in params_to_indx.items()}
    indx_to_params = {i: p for p, i in params_to_indx.items()}
    return params_to_indx, indx_to_params
    
class HeparinDataset(object):
    
    # TODO: Find a way not to hard code these in here?
    raw_data_filename = 'cotesting_updated_v7.csv'
    pat_id_col = 'ANON_ID'
    time_col = 'Date'
    obs_cluster_idx_col = 'ClusterPosition38'
    params = ['Age', 'PTT', 'HAL', 'PT', 'INR', 'TBILI', 'CR']
    
    params_dict = {k: i for i, k in enumerate(params)}  # param name <-> col indx
    pat_trajs = []  # List of all disjoint patient trajectories
    pat_ids = set()  # List of unique patient ID's
    pat_id_to_traj_ids = defaultdict(set)  # Map patient ID <-> set of traj IDs
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, 
                 root,  # probably 'data/heparin'
                 train=True, 
                 build_from_file=False,
                 quantization=1.0, 
                 n_samples=None, 
                 device=torch.device("cpu")):
        
        self.root = root
        self.train = train
        self.reduce = "average"
        self.quantization = quantization
        
        if build_from_file:
            self.build_from_file()
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found. Try building from file (.csv)')
        
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, data_file), 
                                   map_location='cpu')
            # self.labels = torch.load(os.path.join(self.processed_folder, self.label_file), 
            #                          map_location='cpu')  # TODO: Handle label files
        else:
            self.data = torch.load(os.path.join(self.processed_folder, data_file))
            # self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))
            # TODO: Handle label files
        
        if not n_samples:
            self.data = self.data[:n_samples]
            # self.labels = self.labels[:n_samples]
    
    
    def build_from_file(self):
        if self._check_exists():
            return
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # os.makedirs(self.raw_folder, exist_ok=True)  # We assume the raw data exists
        os.makedirs(self.processed_folder, exist_ok=True)
        
        traj_id = 0
        with open(os.path.join(self.raw_folder, self.raw_data_filename)) as f:
            prev_line = None  # Consider making this into a trajectory object
            t_traj_start = None
            prev_t = 0.
            tt = [prev_t]
            vals = [torch.zeros(len(self.params)).to(self.device)]
            mask = [torch.zeros(len(self.params)).to(self.device)]
            nobs = [torch.zeros(len(self.params))]
            labels = None
            traj_is_empty = True
            
            for i, line in tqdm(enumerate(f)):

                # We parse the first line to get map of params <-> column indxs
                if i == 0:
                    params_to_indx, indx_to_params = parse_header(line)
                    continue

                # For all of the data (non-header lines)...
                else:
                    vals_from_line = [str.replace(s, '"', '') for s in line.split(',')]

                    obs_idx = int(vals_from_line[params_to_indx[self.obs_cluster_idx_col]])

                    # We map all times so that 0 marks the first obs.
                    if obs_idx == 1:
                        if not traj_is_empty:
                            tt = torch.tensor(tt).to(self.device)
                            vals = torch.stack(vals)
                            mask = torch.stack(mask)
                            self.pat_trajs.append((traj_id, tt, vals, mask, labels))
                            traj_id += 1
                        
                        # Create a new trajectory
                        prev_line = None
                        prev_t = 0.
                        tt = [prev_t]
                        vals = [torch.zeros(len(self.params)).to(self.device)]
                        mask = [torch.zeros(len(self.params)).to(self.device)]
                        nobs = [torch.zeros(len(self.params))] 
                        pat_id = int(vals_from_line[params_to_indx[self.pat_id_col]])
                        self.pat_ids.add(pat_id)
                        self.pat_id_to_traj_ids[pat_id].add(traj_id)
                        traj_is_empty = True

                        tmp_t = vals_from_line[params_to_indx[self.time_col]]
                        tmp_t = datetime.strptime(tmp_t, '%Y-%m-%d %H:%M:%S')
                        t_traj_start = tmp_t

                    t = vals_from_line[params_to_indx[self.time_col]]
                    t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                    t = t - t_traj_start
                    t = t.seconds / 60. / 60. + t.days * 24
                    t = round(t / self.quantization) * self.quantization

                    # If we're still in the same trajectory and not aggregating 
                    # within a single time window, then we add an entry to vals etc.
                    if t != prev_t:
                        tt.append(t)
                        vals.append(torch.zeros(len(self.params)).to(self.device))
                        mask.append(torch.zeros(len(self.params)).to(self.device))
                        nobs.append(torch.zeros(len(self.params)))
                        prev_t = t

                    for param in self.params:
                        n_observations = nobs[-1][self.params_dict[param]]
                        val_from_line = vals_from_line[params_to_indx[param]]
                        if val_from_line != 'NA':
                            val = float(val_from_line)
                            if self.reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        traj_is_empty = False
            
            # Make sure you include the last trajectory in the set of trajectories
            if not traj_is_empty:
                tt = torch.tensor(tt).to(self.device)
                vals = torch.stack(vals)
                mask = torch.stack(mask)
                self.pat_trajs.append((traj_id, tt, vals, mask, labels))
                traj_id += 1

            train_trajs, test_trajs = self._get_train_test_split(self.pat_ids, 
                                                                 self.pat_id_to_traj_ids,
                                                                 self.pat_trajs)


            for trajs, split_name in zip([train_trajs, test_trajs], 
                                         ['train', 'test']):
                torch.save(
                    trajs,
                    os.path.join(
                        self.processed_folder,
                        split_name + '_' + str(self.quantization) + '.pt'
                    )
                )
            
            print('Done!')
    
    
    def _check_exists(self):
        for split in ['train', 'test']:
            if not os.path.exists(
                os.path.join(
                    self.processed_folder, 
                    split + '_' + str(self.quantization) + '.pt'
                )
            ):
                return False
        return True
    
    
    def _parse_header(self, line):
        params_in_file = line.split(',')
        params_in_file = [str.replace(s, '"', '').strip() for 
                          s in params_in_file]
        params_to_indx = {p: i for i, p in enumerate(params_in_file)}
        params_to_indx = {p: i for p, i in params_to_indx.items()}
        indx_to_params = {i: p for p, i in params_to_indx.items()}
        return params_to_indx, indx_to_params
    
    
    def _get_train_test_split(self, 
                              pat_ids, 
                              pat_id_to_traj_ids, 
                              pat_trajs, 
                              seed=42):
        random.seed(seed)
        
        train_pat_ids = random.sample(pat_ids, int(0.7 * len(pat_ids)))
        train_traj_ids = []
        for pat_id in train_pat_ids:
            for traj_id in pat_id_to_traj_ids[pat_id]:
                train_traj_ids.append(traj_id)
                 
        test_pat_ids = [pid for pid in pat_ids if pid not in train_pat_ids]
        test_traj_ids = []
        for pat_id in test_pat_ids:
            for traj_id in pat_id_to_traj_ids[pat_id]:
                test_traj_ids.append(traj_id)
        
        train_trajs = [pat_trajs[i] for i in train_traj_ids]
        test_trajs = [pat_trajs[i] for i in test_traj_ids]
                 
        assert len(set(train_pat_ids).intersection(set(test_pat_ids))) == 0
        assert len(set(train_traj_ids).intersection(set(test_traj_ids))) == 0
        
        assert set([traj[0] for traj in train_trajs]) == set(train_traj_ids)
        assert set([traj[0] for traj in test_trajs]) == set(test_traj_ids)
        
        return train_trajs, test_trajs
    
    
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')
    
    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')
    
    @property
    def training_file(self):
        return 'train_' + str(self.quantization) + '.pt'
    
    @property
    def test_file(self):
        return 'test_' + str(self.quantization) + '.pt'
    
    # TODO: Handle the outcomes for the heparin dataset
    # @property
    # def label_file(self):
    #     return 'Outcomes-a.pt'
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def get_label(self, record_id):
        return self.labels[record_id]
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format('train' if self.train is True else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Quantization: {}\n'.format(self.quantization)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str
    
    def visualize(self, timesteps, data, mask, plot_name):
        width = 15
        height = 15

        non_zero_attributes = (torch.sum(mask,0) > 2).numpy()
        non_zero_idx = [i for i in range(len(non_zero_attributes)) if non_zero_attributes[i] == 1.]
        n_non_zero = sum(non_zero_attributes)
        
        mask = mask[:, non_zero_idx]
        data = data[:, non_zero_idx]

        params_non_zero = [self.params[i] for i in non_zero_idx]
        params_dict = {k: i for i, k in enumerate(params_non_zero)}
        
        n_col = 3
        n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
        fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height), facecolor='white')

        #for i in range(len(self.params)):
        for i in range(n_non_zero):
            param = params_non_zero[i]
            param_id = params_dict[param]

            tp_mask = mask[:,param_id].long()

            tp_cur_param = timesteps[tp_mask == 1.]
            data_cur_param = data[tp_mask == 1., param_id]
            
            # TODO: If you only have 3 labs, then you'll have a 1D array which will throw error
            ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(), data_cur_param.numpy(),  marker='o') 
            ax_list[i // n_col, i % n_col].set_title(param)

        fig.tight_layout()
        fig.savefig(plot_name)
        plt.close(fig)
    

def variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
    data_min = None, data_max = None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
        - record_id is a patient id
        - tt is a 1-dimensional tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    D = batch[0][2].shape[1]
    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), 
                                                sorted=True, 
                                                return_inverse=True)
    combined_tt = combined_tt.to(device)
    
    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    
    combined_labels = None
    N_labels = 1
    
    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
    combined_labels = combined_labels.to(device = device)
    
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)
        
        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        if labels is not None:
            combined_labels[b] = labels

    combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, 
        att_min = data_min, att_max = data_max)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {
        "data": combined_vals, 
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": combined_labels}

    data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
    return data_dict
    
    
if __name__ == '__main__':
    torch.manual_seed(42)
    dataset = HeparinDataset(root='../data/heparin', train=True, build_from_file=True)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=variable_time_collate_fn)
    print(dataloader.__iter__().next())