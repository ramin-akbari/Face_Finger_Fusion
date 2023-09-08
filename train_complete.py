import argparse
import pickle
import re
import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, distances
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAUROC, BinaryROC
from tqdm import trange


class FaceFingerData:
    def __init__(self, face_adr='face.pkl', finger_adr='fingerprint.pkl'):
        def clear_str(reg_pat: str, x) -> str:
            return re.search(reg_pat, x).group()

        with open(face_adr, 'rb') as file:
            face_pkl = pickle.load(file)
            self.face_feat = np.array(face_pkl['feats'], dtype=np.float32)
            self.face_df = pd.DataFrame({'id': map(lambda x: clear_str(r'^[\d]+(?=_)', x), face_pkl['fnames'])})
            self.face_df['index'] = range(len(self.face_df))

        with open(finger_adr, 'rb') as file:
            finger_pkl = pickle.load(file)
            self.finger_feat = np.array(finger_pkl['feats'], dtype=np.float32)
            self.finger_df = pd.DataFrame(
                {'id': map(lambda x: clear_str(r'(?<=\/)[0-9]*(?=_)', x), finger_pkl['fnames'])})
            self.finger_df['index'] = range(len(self.finger_df))

    def fuse_data(self, label_threshhold=5, show_report=False):
        df = self.face_df.merge(self.finger_df, on='id', how='inner')
        df['repetition'] = df[['id', 'index_x']].groupby('id').transform(len)
        df = df[df['repetition'].gt(label_threshhold)]
        df['label'] = df.groupby('id', sort=False).ngroup()
        selected_faces = self.face_feat[df['index_x'].values]
        selected_fingers = self.finger_feat[df['index_y'].values]
        fused = np.concatenate((selected_faces, selected_fingers), axis=1)

        if show_report:
            print(f"face features were :        \t {self.face_feat.shape}")
            print(f"finger print features were :\t {self.finger_feat.shape}")
            print(f"fused features are :        \t {fused.shape}")
            print(f"number of classes are  :    \t {df['label'].values[-1]}")

        return fused, df['label'].values, fused.shape[1]


def split_data(data_x, data_label, test_p=0.2, noise_level=None):
    n_label = np.max(data_label)
    n_test = int(test_p * n_label)
    test_index = np.random.choice(n_label + 1, n_test, replace=False)
    test_index = np.isin(data_label, test_index)

    x_train = data_x[~test_index]
    label_train = data_label[~test_index]
    x_val = data_x[test_index]
    label_val = data_label[test_index]

    if noise_level is not None:
        train_noise = np.random.choice(x_train.shape[0], int(x_train.shape[0] * 0.2), replace=False)
        x_train[train_noise] += noise_level * (np.random.randn(int(x_train.shape[0] * 0.2), x_train.shape[1]))

        val_noise = np.random.choice(x_val.shape[0], int(x_val.shape[0] * 0.2), replace=False)
        x_val[val_noise] += noise_level * (np.random.randn(int(x_val.shape[0] * 0.2), x_val.shape[1]))

    return (x_train, label_train), (x_val, label_val)


def score_filter(X):
    return torch.sigmoid(5 * X - 1.5) - torch.sigmoid(7 * X - 16)
    # return torch.sigmoid(X)


def dot_filter(X):
    return torch.tanh(2 * X) - 2 * torch.sigmoid(7 * X - 16)
    # return torch.tanh(X)


class BiometricDataSet(Dataset):
    def __init__(self, x, label):
        super().__init__()
        self.x = x
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.x[index], self.label[index]


class BiometricProjector():
    def __init__(self, feature_dim, projected_dim, lmb, margin, train_data, test_data, batch_size=512, p_dropout=0.45,
                 seed_value=0, use_filter=False, device=torch.device("cpu")):
        torch.manual_seed(seed_value)
        x_train = torch.from_numpy(train_data[0]).to(device)
        label_train = torch.from_numpy(train_data[1]).to(device)
        x_test = torch.from_numpy(test_data[0]).to(device)
        label_test = torch.from_numpy(test_data[1]).to(device)
        self.train_data = DataLoader(BiometricDataSet(x_train, label_train), batch_size=batch_size, shuffle=True,
                                     drop_last=True)
        self.test_data = (x_test, label_test)
        self.dim = projected_dim
        self.use_filter = use_filter
        self.lmb = lmb
        self.margin = margin
        self.dropout = nn.Dropout(p=p_dropout)
        self.fc1 = nn.Linear(feature_dim, projected_dim, bias=False, device=device)
        self.loss = TripletLoss(self.margin, batch_size, device, self.use_filter)
        self.auroc = AUROC(self.test_data[1], device, self.use_filter)
        self.optimizer = torch.optim.Adam(self.fc1.parameters(), lr=lr, weight_decay=regularization)

        self.auroc_init = self.metric_validation()

    def _forward(self, x):
        return self.fc1(self.dropout(x))

    def _eval(self, x):
        return self.fc1(x)

    @torch.no_grad()
    def prd_validation(self):
        return self._eval(self.test_data[0])

    def metric_validation(self):
        score = self.auroc.get_score(self.prd_validation())
        return self.auroc(score)

    def train(self):
        avg_loss = 0
        for x, label in self.train_data:
            prd = self._forward(x)
            loss = self.loss(prd, label, self.lmb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.detach()
        return avg_loss / len(self.train_data)

    def report(self):
        projection_score = self.auroc.get_score(self.prd_validation())
        score_tuple = self.auroc.decompose(projection_score)

        if self.use_filter:
            concat_score = self.auroc.sigmoid_score(self.test_data[0])
        else:
            concat_score = self.auroc.get_score(self.test_data[0])

        concat_auroc = self.auroc(concat_score).cpu()
        projection_auroc = self.auroc(projection_score)
        roc_x, roc_y = self.auroc.get_roc_data(projection_score)

        cdf_data = []
        for score in score_tuple:
            pdf, xbin = np.histogram(score.cpu().numpy(), bins=256, density=True)
            dx = xbin[1] - xbin[0]
            x = (xbin[:-1] + xbin[1:]) * 0.5
            cdf = np.cumsum(pdf)
            cdf *= dx
            cdf_data.extend([x, cdf])

        out = [concat_auroc, self.auroc_init, projection_auroc, roc_x, roc_y]
        out = [obj.cpu().numpy() for obj in out]
        out.extend(cdf_data)
        return out

    def __repr__(self) -> str:
        return (f"{'filter_based' if self.use_filter else 'exact'}_lmb:{self.lmb:<4.2f}"
                f"_mrgn:{self.margin:<4.2f}_{self.dim:d}")


class TanhSimilarity(distances.DotProductSimilarity):
    def __init__(self, **kwargs):
        super().__init__(normalize_embeddings=False, **kwargs)

    def compute_mat(self, query_emb, ref_emb):
        return dot_filter(super().compute_mat(query_emb, ref_emb))

    def pairwise_distance(self, query_emb, ref_emb):
        return dot_filter(super().pairwise_distance(query_emb, ref_emb))


class TripletLoss:
    def __init__(self, margin, batch_size, device, use_filter):
        self.row, self.col = torch.triu_indices(batch_size, batch_size, offset=1, device=device)
        if use_filter:
            self.push = losses.TripletMarginLoss(margin=margin, distance=TanhSimilarity())
            self.pull = self._pull_filter
        else:
            self.push = losses.TripletMarginLoss(margin=margin, distance=distances.CosineSimilarity())
            self.pull = self._pull_exact

    def __call__(self, x: torch.tensor, label: torch.tensor, lmb):
        mask = (label.unsqueeze(0) == label.unsqueeze(1))[self.row, self.col]
        pull = self.pull(x, mask)
        pull = 1 - torch.sum(pull) / mask.sum()
        push = self.push(x, label)
        return push + lmb * pull

    def _same_class_dot(self, x, mask):
        return torch.sum(x[self.row[mask]] * x[self.col[mask]], axis=1)

    def _pull_filter(self, x, mask):
        return dot_filter(self._same_class_dot(x, mask))

    def _pull_exact(self, x, mask):
        x = x / x.norm(dim=1).unsqueeze(1)
        return self._same_class_dot(x, mask)


class AUROC:
    def __init__(self, val_label, device, use_filter, thresholds=256):
        self.row, self.col = torch.triu_indices(len(val_label), len(val_label), offset=1, device=device)
        self.true_label = (val_label.unsqueeze(0) == val_label.unsqueeze(1))[self.row, self.col]
        self.metric = BinaryAUROC(thresholds=thresholds).to(device)
        self.metric_curve = BinaryROC(thresholds=thresholds).to(device)
        if use_filter:
            self.get_score = self._filter_score
        else:
            self.get_score = self._exact_score

    def __call__(self, score):
        return self.metric(score, self.true_label)

    def _gram_matrix_triu(self, x):
        return (x @ x.T)[self.row, self.col]

    def _exact_score(self, x):
        x = x / x.norm(dim=1).unsqueeze(1)
        return self._gram_matrix_triu(x)

    def _filter_score(self, x):
        return score_filter(self._gram_matrix_triu(x))

    def sigmoid_score(self, x):
        return torch.sigmoid(self._gram_matrix_triu(x))

    def decompose(self, score):
        score_p = score[self.true_label]
        score_n = score[~self.true_label]
        return score_p, score_n

    def get_roc_data(self, score):
        x_values, y_values, _ = self.metric_curve(score, self.true_label)
        return x_values, y_values


class Logger:
    def __init__(self,filter_based=False,noise_level=None):

        folder_tree = ["Logs","FilterBased" if filter_based else "Exact","Raw" if noise_level is None else "Noisy"]
        curr_folder = ''
        for folder in folder_tree:
            curr_folder = os.path.join(curr_folder,folder)
            if not os.path.exists(curr_folder):
                os.mkdir(curr_folder)

        file_name = time.strftime("%h%d%H%M%S")
        out_path = os.path.join(curr_folder, file_name + ".csv")
        duplicate = 0
        while os.path.isfile(out_path):
            duplicate += 1
            out_path = os.path.join("Logs", f"{file_name}_{duplicate:d}" + ".csv")

        self.path = out_path
        self.elements = ['model', 'lambda', 'margin', 'proj_dim', 'AUROC_Concat', 'AUROC_Init', 'AUROC',
                         'ROC_X', 'ROC_Y', 'pos_cdf_x','pos_cdf_y', 'neg_cdf_x','neg_cdf_y']
        self.df = pd.DataFrame(columns=self.elements)
        self.df.to_csv(self.path, index=False)
        self.max_length = 100

    def save(self, model_obj: BiometricProjector):
        if len(self.df) == self.max_length:
            self.dump()

        curr_index = len(self.df)
        log_list = [repr(model_obj), model_obj.lmb.item(), model_obj.margin.item(), model_obj.dim.item(), *model_obj.report()]
        self.df.loc[curr_index, :] = log_list

    def dump(self):
        self.df.to_csv(self.path, index=False, header=False,mode='a')
        self.df = pd.DataFrame(columns=self.elements)


def process_args(args: dict[str, str]):
    out = []
    default_args = {'lambda':torch.tensor([0.05,0.1,0.2,0.5,1]),
                    'margin':torch.tensor([0.1,0.2,0.3,0.35,0.4,0.45,0.5,0.6,0.7]),
                    'pdims': [32,64,128,256]}
    special_args = {'gpu' : lambda x : 'cpu' if x is None else x,
                    'filter' : lambda x : False if x is None else True if x.lower()=='true' else False,
                    'noise' : lambda x : None if x is None else float(x)}
    for key in args:
        if key not in special_args:
            if args[key] is not None:
                var = re.split('[\[\]()\',]', args[key])
                var = filter(lambda s: s != '', var)
                var = list(map(float, var))
                if key=='pdims':
                    var = torch.tensor(var,dtype=torch.int64)
                elif len(var) == 2:
                    var = torch.linspace(var[0], var[1], 2)
                elif len(var) == 3:
                    var = torch.linspace(var[0], var[1], int(var[2]))
                else:
                    var = torch.tensor(var)
            else:
                var = default_args[key]          
        else:
            var = special_args[key](args[key])
        out.append(var)

    return out

np.random.seed(50)
torch.manual_seed(1234)
parser = argparse.ArgumentParser()
parser.add_argument("--lambda", help="Network lambda hyper parameter")
parser.add_argument("--margin", help="Network margin hyper parameter")
parser.add_argument("--pdims",  help="Network Projection Dimension")
parser.add_argument("--filter", help="Use Filter Based Training")
parser.add_argument("--noise",  help="Use Filter Based Training")
parser.add_argument("--gpu", help="gpu device number ex = cuda:0 or cuda:1")
arg_dict = vars(parser.parse_args())

lmb_vec, margin_vec, pdims, use_filter, noise_level, device = process_args(arg_dict)

print(use_filter)

device = torch.device(device)
torch.set_num_threads(32)

repetition  = 8
batch_size = 256
lr = 1e-4
regularization = 5e-4
epochs = 400

data = FaceFingerData()
fused_feature, label, in_dim = data.fuse_data(label_threshhold=8)
logger = Logger(use_filter,noise_level)


for lmb in lmb_vec:
    for margin in margin_vec:
        for pdim in pdims:
            np.random.seed(0)
            for rep in range(repetition):
                train_data, test_data = split_data(fused_feature, label, test_p=0.2,noise_level=noise_level)
                model = BiometricProjector(in_dim, pdim, lmb, margin, train_data, test_data, batch_size=batch_size,
                                           p_dropout=0.45, seed_value=0, use_filter=use_filter, device=device)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(model.optimizer, gamma=0.985)
                p_bar = trange(epochs)
                for index, epoch in enumerate(p_bar):
                    avg_loss = model.train()
                    if not (1 + index) % 5:
                        p_bar.set_description(
                            f"(l:{lmb:4.2f},m:{margin:4.2f},d:{pdim:<3d}) rep:{rep:<1d} loss:{avg_loss :<6.4f}  "
                            f"AUROC:{model.metric_validation():<6.4f}")
                    scheduler.step()
                logger.save(model)
                
logger.dump()