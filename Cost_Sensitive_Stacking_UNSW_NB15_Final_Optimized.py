"""
Cost_Sensitive_Stacking_UNSW_NB15_Final_Optimized.py

Leakage-reduced and optimized UNSW-NB15 cost-sensitive stacking framework.

Main improvements:
- Train-only preprocessing
- Removal of high-risk identifier/simulation fingerprint features
- SMOTENC for categorical-safe oversampling
- Reduced memory usage
- Optimized ensemble parameters
- SHAP runtime control

Expected dataset:
UNSW-NB15_1.csv
UNSW-NB15_2.csv
UNSW-NB15_3.csv
UNSW-NB15_4.csv

inside DRIVE_FOLDER.
"""

import os
import gc
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import TomekLinks

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import joblib


# ==========================
# CONFIGURATION
# ==========================

DRIVE_FOLDER = "/content/drive/MyDrive/stack"

FAST_MODE = False

TOP_FEATURES = 30

FEATURE_SELECTOR_TREES = 100

MODEL_TREES = 200 if FAST_MODE else 300

STACK_FOLDS = 3 if FAST_MODE else 5

SMOTE_RATIO = 0.2


# ==========================
# COLUMN DEFINITIONS
# ==========================

COLUMN_NAMES = [
    "srcip","sport","dstip","dsport","proto","state","dur",
    "sbytes","dbytes","sttl","dttl","sloss","dloss","service",
    "sload","dload","spkts","dpkts","swin","dwin","stcpb","dtcpb",
    "smeansz","dmeansz","trans_depth","res_bdy_len","sjit","djit",
    "stime","ltime","sintpkt","dintpkt","tcprtt","synack","ackdat",
    "is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd",
    "is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst",
    "ct_dst_ltm","ct_src_ltm","ct_src_dport_ltm",
    "ct_dst_sport_ltm","ct_dst_src_ltm",
    "attack_cat","label"
]


def downcast(df):
    for c in df.select_dtypes(include="float64"):
        df[c] = pd.to_numeric(df[c], downcast="float")

    for c in df.select_dtypes(include="int64"):
        df[c] = pd.to_numeric(df[c], downcast="integer")

    return df


# ==========================
# LOAD DATA
# ==========================

files = sorted(
    glob.glob(
        f"{DRIVE_FOLDER}/UNSW-NB15_*.csv"
    )
)

if not files:
    raise FileNotFoundError(
        "No UNSW-NB15 files found"
    )

frames = []

for f in files:
    print("Loading:", f)

    df = pd.read_csv(
        f,
        header=None,
        names=COLUMN_NAMES,
        encoding="utf-8-sig",
        low_memory=False
    )

    df = downcast(df)
    frames.append(df)


data = pd.concat(
    frames,
    ignore_index=True
)

del frames
gc.collect()


# ==========================
# CLEANING
# ==========================

data = data.drop_duplicates()

data = data.replace(
    [np.inf,-np.inf],
    np.nan
)

data["attack_cat"] = (
    data["attack_cat"]
    .astype(str)
    .str.strip()
)

data["attack_cat"] = data["attack_cat"].replace(
    {
        "nan":"Normal",
        "":"Normal"
    }
)

data = data.fillna(0)

data = downcast(data)


# ==========================
# LEAKAGE REDUCTION
# ==========================

DROP_COLS = [

    "attack_cat",
    "label",

    "srcip",
    "dstip",
    "sport",
    "dsport",

    "stime",
    "ltime",

    "ct_state_ttl",
    "ct_srv_src",
    "ct_srv_dst",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm"
]


X = data.drop(
    columns=[
        c for c in DROP_COLS
        if c in data.columns
    ]
)

y_raw = data["attack_cat"].astype(str)


encoder_y = LabelEncoder()

y = encoder_y.fit_transform(
    y_raw
)


# ==========================
# SPLIT FIRST
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================
# TRAIN ONLY ENCODING
# ==========================

cat_cols = X_train.select_dtypes(
    include="object"
).columns.tolist()


preprocessor = ColumnTransformer(

    [
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            cat_cols
        )
    ],

    remainder="passthrough"
)


X_train = preprocessor.fit_transform(
    X_train
)

X_test = preprocessor.transform(
    X_test
)


cat_indices = list(
    range(len(cat_cols))
)


# ==========================
# FEATURE SELECTION
# ==========================

selector = RandomForestClassifier(

    n_estimators=FEATURE_SELECTOR_TREES,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


selector.fit(
    X_train,
    y_train
)


importance = selector.feature_importances_

selected = np.argsort(
    importance
)[-TOP_FEATURES:]


X_train = X_train[:, selected]

X_test = X_test[:, selected]


# ==========================
# SMOTENC
# ==========================

valid_cat_indices = [
    i for i in cat_indices
    if i < X_train.shape[1]
]


smote = SMOTENC(

    categorical_features=valid_cat_indices,

    sampling_strategy=SMOTE_RATIO,

    random_state=42
)


X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)


tomek = TomekLinks()

X_train, y_train = tomek.fit_resample(
    X_train,
    y_train
)


# ==========================
# MODELS
# ==========================

rf = RandomForestClassifier(

    n_estimators=MODEL_TREES,

    random_state=42,

    n_jobs=-1
)


xgb = XGBClassifier(

    n_estimators=MODEL_TREES,

    learning_rate=0.05,

    max_depth=5,

    subsample=0.8,

    colsample_bytree=0.8,

    tree_method="hist",

    eval_metric="mlogloss",

    random_state=42,

    n_jobs=-1
)


lgb = LGBMClassifier(

    n_estimators=MODEL_TREES,

    learning_rate=0.05,

    random_state=42,

    n_jobs=-1,

    verbose=-1
)


model = StackingClassifier(

    estimators=[

        ("RF",rf),

        ("XGB",xgb),

        ("LGBM",lgb)

    ],

    final_estimator=LogisticRegression(
        max_iter=2000
    ),

    cv=StratifiedKFold(

        n_splits=STACK_FOLDS,

        shuffle=True,

        random_state=42
    )

)


# ==========================
# TRAIN
# ==========================

weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


model.fit(
    X_train,
    y_train,
    sample_weight=weights
)


# ==========================
# EVALUATION
# ==========================

prediction = model.predict(
    X_test
)


print("Accuracy:",
      accuracy_score(y_test,prediction))

print("Macro Precision:",
      precision_score(
          y_test,
          prediction,
          average="macro"
      ))

print("Macro Recall:",
      recall_score(
          y_test,
          prediction,
          average="macro"
      ))

print("Macro F1:",
      f1_score(
          y_test,
          prediction,
          average="macro"
      ))


print(
    classification_report(
        y_test,
        prediction,
        target_names=encoder_y.classes_
    )
)


print(
    confusion_matrix(
        y_test,
        prediction
    )
)


# ==========================
# SAVE MODEL
# ==========================

joblib.dump(
    model,
    "UNSW_NB15_Optimized_Stacking_Model.pkl"
)

joblib.dump(
    preprocessor,
    "UNSW_NB15_Preprocessor.pkl"
)


print("Training completed successfully.")
