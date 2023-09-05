import pickle
import re
import numpy as np
import pandas as pd


def clear_str(reg_pat: str, x) -> str:
    return re.search(reg_pat, x).group()

with open('face.pkl', 'rb') as file:
    face_pkl = pickle.load(file)
    face_feat = np.array(face_pkl['feats'])
    face_df = pd.DataFrame({'id': map(lambda x: clear_str(r'^[\d]+(?=_)', x), face_pkl['fnames'])})
    face_df['index'] = range(len(face_df))

with open('fingerprint.pkl', 'rb') as file:
    finger_pkl = pickle.load(file)
    finger_feat = np.array(finger_pkl['feats'])
    finger_df = pd.DataFrame({'id': map(lambda x: clear_str(r'(?<=\/)[0-9]*(?=_)', x), finger_pkl['fnames'])})
    finger_df['index'] = range(len(finger_df))

df = face_df.merge(finger_df, on='id', how='inner')

df['label'] = df.groupby('id', sort=False).ngroup()

selected_faces = face_feat[df['index_x'].values]
selected_fingers = finger_feat[df['index_y'].values]

fused = np.concatenate((selected_faces, selected_fingers), axis=1)

print(f"face features were :        \t {face_feat.shape}")
print(f"finger print features were :\t {finger_feat.shape}")
print(f"fused features are :        \t {fused.shape}")
print(f"number of classes are  :    \t {df['label'].values[-1]}")

np.savetxt("fused_feat.txt",fused, comments='')
np.savetxt("labels.txt", df['label'].values, fmt='%d', comments='')
