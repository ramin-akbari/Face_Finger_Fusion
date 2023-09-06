import argparse
import re
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, distances
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAUROC, BinaryROC
from tqdm import trange


def get_data(device=torch.device("cpu"), test_prcntg=0.2):
    data_x = np.loadtxt('fused_feat.txt', dtype=np.float32)
    data_label = np.loadtxt('labels.txt',dtype=np.int64)

    feature_dim = data_x.shape[1]
    n_label = np.max(data_label)
    n_test = int(test_prcntg * n_label)
    test_index = np.random.choice(n_label + 1,n_test, replace=False)
    test_index = np.isin(data_label, test_index)

    x_train = data_x[~test_index]
    label_train = data_label[~test_index]
    x_val = data_x[test_index]
    label_val = data_label[test_index]

    noise_level = 0.075
    train_noise = np.random.choice(x_train.shape[0],x_train.shape[0]//2, replace=False)
    x_train[train_noise] += noise_level*np.random.randn(x_train.shape[0]//2 ,x_train.shape[1])
    
    val_noise = np.random.choice(x_val.shape[0],x_val.shape[0]//2, replace=False)
    x_val[val_noise] += noise_level*np.random.randn(x_val.shape[0]//2 ,x_val.shape[1])

    x_train = torch.from_numpy(x_train).to(device)
    x_val = torch.from_numpy(x_val).to(device)
    label_train = torch.from_numpy(label_train).to(device)
    label_val = torch.from_numpy(label_val).to(device)


    return (x_train, label_train), (x_val, label_val), feature_dim

def score_filter(X):
    return torch.sigmoid(8 * X - 2) - torch.sigmoid(7 * X - 15)
    # return torch.sigmoid(X)

def dot_filter(X):
    return torch.tanh(2.6*X) - torch.sigmoid(7 * X - 15)
    # return torch.sigmoid(X)

class BiometricDataSet(Dataset):
    def __init__(self, x, label):
        super().__init__()
        self.x = x
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.x[index], self.label[index]

class BiometricProjector(nn.Module):
    def __init__(self, feature_dim, projected_dim, is_linear=True,drop_out_p = 0.3, device=torch.device("cpu")):
        torch.manual_seed(1234)
        super().__init__()
        self.linear = is_linear
        self.dim = (feature_dim, projected_dim)
        self.p = drop_out_p
        self.drop_out = nn.Dropout(p=self.p)
        if self.linear:
            self.fc1 = nn.Linear(feature_dim, projected_dim, bias=False, device=device)
            self.compute = self._linear_compute
        else:
            self.fc1 = nn.Linear(feature_dim, feature_dim, bias=True, device=device)
            self.ac1 = nn.SiLU()
            self.fc2 = nn.Linear(feature_dim, projected_dim, bias=False, device=device)
            self.compute = self._nonlinear_compute

    def forward(self, x):
        return self.compute(x)
    
    def _linear_compute(self,x):
        return self.fc1(self.drop_out(x))

    def _nonlinear_compute(self,x):
        return self.fc2(self.drop_out(self.ac1(self.fc1(x))))
    
    def __repr__(self):
         if self.linear:
             return f"Linear Network - DropOut: {self.p:<5.2f}%  > {self.dim[0]:>5d} > {self.dim[1]:<5d}"
         else:
             return f"Shallow Network {self.dim[0]:>5d} > SiLU > DropOut: {self.p:<5.2f}% > {self.dim[1]:<5d}"
        
class TanhSimilarity(distances.DotProductSimilarity):
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=False, **kwargs)

    def compute_mat(self, query_emb, ref_emb):
        return dot_filter(super().compute_mat(query_emb, ref_emb))

    def pairwise_distance(self, query_emb, ref_emb):
        return dot_filter(super().pairwise_distance(query_emb, ref_emb))

class TripletLoss:
    def __init__(self, batch_size, lmb, margin, device, use_filter):
        self.row, self.col = torch.triu_indices(batch_size, batch_size, offset=1, device=device)
        self.lmb = lmb
        if use_filter:
            self.push = losses.TripletMarginLoss(margin=margin, distance=TanhSimilarity())
            self.pull = self._pull_filter
        else:
            self.push = losses.TripletMarginLoss(margin=margin, distance=distances.CosineSimilarity())
            self.pull = self._pull_exact

    def __call__(self, x: torch.tensor, label: torch.tensor):
        mask = (label.unsqueeze(0) == label.unsqueeze(1))[self.row, self.col]
        pull = self.pull(x,mask)
        pull = 1 - torch.sum(pull) / mask.sum()
        push = self.push(x, label)
        return push + self.lmb * pull
    
    def _pull_filter(self,x,mask):
        pull = torch.sum(x[self.row[mask]] * x[self.col[mask]], axis=1)
        pull = score_filter(pull)
        return pull
    
    def _pull_exact(self,x,mask):
        x = x / x.norm(dim=1).unsqueeze(1)
        pull = torch.sum(x[self.row[mask]] * x[self.col[mask]], axis=1)
        return pull

class AUROC:
    def __init__(self, val_label, device,use_filter ,thresholds=512):
        self.row, self.col = torch.triu_indices(len(val_label), len(val_label), offset=1, device=device)
        self.true_label = (val_label.unsqueeze(0) == val_label.unsqueeze(1))[self.row, self.col]
        self.metric = BinaryAUROC(thresholds=thresholds)
        self.metric_curve = BinaryROC(thresholds=thresholds)
        self.filter = use_filter
        if self.filter:
            self.get_score = self._filter_score
        else:
            self.get_score = self._exact_score

    def compute(self, x):
        score = self.get_score(x)
        return self.metric(score, self.true_label)
    
    def _exact_score(self,x):
        x = x / x.norm(dim=1).unsqueeze(1)
        score = (x @ x.T)[self.row, self.col]
        return score
    
    def _filter_score(self,x):
        score = (x @ x.T)[self.row, self.col]
        score = score_filter(score)
        return score

    @torch.no_grad()
    def get_report(self, x, hyper_param, model_obj):
        if self.filter:
            x_init = (x @ x.T)[self.row, self.col]
            score_init = torch.sigmoid(x_init)
            sub_folder = 'FilterBased'
        else:
            score_init = self.get_score(x)
            sub_folder = 'Exact'
            
        x_net = model_obj(x)
        score_net = self.get_score(x_net)

        score_data = [score_init,score_net]

        json_path,plot_path = get_path(hyper_param, sub_folder,model_obj.linear)
        fig, ax = plt.subplots(2, 2)
        fig.set_dpi(100)
        fig.set_size_inches(20, 20)

        for i,score in enumerate(score_data):
            score_p = score[self.true_label]
            score_n = score[~self.true_label]

            ax[i,0].hist(score_n.cpu().numpy(), alpha=0.7, bins=1000, density=True, cumulative=True)
            ax[i,0].hist(score_p.cpu().numpy(), alpha=0.6, bins=1000, density=True, cumulative=True)
            if self.filter:
                ax[i,0].set_xlim([-0.05, 1.05])
                ax[i,0].set_xticks(np.linspace(0,1,11))
            else:
                ax[i,0].set_xlim([-1.05, 1.05])
                ax[i,0].set_xticks(np.linspace(-1,1,11))

            ax[i,0].set_ylim([-0.05, 1.05])
            ax[i,0].set_yticks(np.linspace(0,1,11))
            roc_curve = self.metric_curve(score, self.true_label)
            self.metric_curve.plot(roc_curve, ax=ax[i,1])

            ax[i,1].tick_params(axis='both', which='major', labelsize=15)
            ax[i,1].set_title("Binary ROC",fontsize=17)
            ax[i,1].set_xlabel('Flase Positive Rate',fontsize=17)
            ax[i,1].set_ylabel('True Positive Rate',fontsize=17)
            ax[i,1].set_xticks(np.linspace(0,1,11))
            ax[i,1].set_yticks(np.linspace(0,1,11))
            text_kwargs = dict(ha='center', va='center', fontsize=18, color='black')
            ax[i,1].text(0.75, 0.25, f'AUROC = {self.metric(score, self.true_label):6.4f}', **text_kwargs)

            ax[i,0].tick_params(axis='both', which='major', labelsize=15)
            ax[i,0].set_title("CDF of True/False Labels",fontsize=17)
            ax[i,0].grid(visible=True, which='major')


        with open(json_path, 'w') as file:
            auroc_val = self.metric(score_net, self.true_label).item()
            out = {'auroc':auroc_val, 
                   'lambda':hyper_param[0].item(),
                   'margin':hyper_param[1].item(),
                   'Model':repr(model_obj)}
            json.dump(out,file)

        fig.suptitle(f"{'Filter Based Model:' if self.filter else 'Exact Model:'} {repr(model_obj)}"
                     f"(lambda={hyper_param[0]:<4.2f} , margin={hyper_param[1]:<4.2f})",fontsize=18)
        fig.tight_layout(pad=3)
        fig.savefig(plot_path)

        plt.close(fig)

def get_path(hyper_param, branch, linear_model):

    folder_list = ["Result",branch]
    if linear_model:
        folder_list.append("Linear")
    else:
        folder_list.append("NonLinear")

    save_folder = ''
    for name in folder_list:
        save_folder = os.path.join(save_folder,name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    extension = [".json",".png"]
    folder_name = ["Json","Plots"]
    file_name = f"{folder_list[-1]}_lambda_{hyper_param[0]:.2f}_margin_{hyper_param[1]:.2f}"
    
    from string import ascii_lowercase as alphabet
    suffix_list = [f"_{char}" for char in alphabet]
    suffix_list = ['', *suffix_list]
    
    out_path = []
    for ext,name in zip(extension,folder_name):
        folder = os.path.join(save_folder,name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        failed_file = True
        for suffix in suffix_list:
            out_name = file_name + suffix + ext
            out_file = os.path.join(folder,out_name)
            if not os.path.isfile(out_file):
                failed_file = False
                out_path.append(out_file)
                break
        if failed_file:
            raise Exception("You are runnig the same model for more than 26 times, time to think !!")

    return out_path

def process_args(args: dict[str, str], omitted=None):
    if omitted is None:
        omitted = {}
    out = []
    for key in args:
        if key not in omitted:
            if args[key] is None:
                raise Exception(f"{key} can not be None. "
                                f"Define command-line argument as:\n"
                                f"--{key}='(lower,upper,npoints)'")
            var = re.split('[\[\]()\',]', args[key])
            var = filter(lambda s: s != '', var)
            var = list(map(float, var))
            if len(var) == 2:
                var = torch.linspace(var[0], var[1], 2)
            elif len(var) == 3:
                var = torch.linspace(var[0], var[1], int(var[2]))
            else:
                var = torch.tensor(var)
        else:
            var = args[key]
        out.append(var)

    return out

np.random.seed(1234)
torch.manual_seed(1234)
parser = argparse.ArgumentParser()
parser.add_argument("--lambda", help="Network lambda hyper parameter")
parser.add_argument("--margin", help="Network margin hyper parameter")
parser.add_argument("--gpu", help="gpu device number ex = cuda:0 or cuda:1")
arg_dict = vars(parser.parse_args())

lmb_vec, margin_vec, gpu = process_args(arg_dict, {'gpu'})

device = torch.device(gpu if gpu is not None else "cpu")
torch.set_num_threads(64)

linear_model = True
use_filter = False

out_dim = 32
batch_size = 256

lr = 2e-4
regularization = 1e-5
epochs = 500

data_train, data_val, in_dim = get_data(device=device)
data_train = DataLoader(BiometricDataSet(*data_train), batch_size=batch_size, shuffle=True, drop_last=True)
auroc = AUROC(data_val[1], device, use_filter)


for lmb in lmb_vec:
    for margin in margin_vec:
        model = BiometricProjector(in_dim, out_dim, device=device,is_linear=linear_model)
        loss_func = TripletLoss(batch_size, lmb, margin,device,use_filter)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        p_bar = trange(epochs)
        
        for epoch in p_bar:
            for index, (x, label) in enumerate(data_train):
                loss = loss_func(model(x), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if not (epoch + 1) % 20:
                with torch.no_grad():
                    model.eval()
                    auroc_val = auroc.compute(model(data_val[0]))
                    model.train()
                    p_bar.set_description(f"lambda:{lmb:<4.2f}  margin:{margin:<4.2f}  "
                                        f"AUROC:{auroc_val:<6.4f}")

            scheduler.step()
        model.eval()
        auroc.get_report(data_val[0], (lmb, margin), model)
        model.train()
