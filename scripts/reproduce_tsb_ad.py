# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import argparse
import pathlib
import random
import time
from datetime import datetime
from typing import Literal, get_args

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from TSB_AD.model_wrapper import (
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool,
    find_length_rank,
    run_Semisupervise_AD,
    run_Unsupervise_AD,
)

_SUBSET_TYPE = Literal["Tuning", "Eva"]


# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    print("CUDA Available: ", torch.cuda.is_available())
    print("cuDNN Version: ", torch.backends.cudnn.version())

    # ArgumentParser
    parser = argparse.ArgumentParser(description="Running TSB-AD")
    # parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')  # Replace with set of time series (Tuning or Eva)
    parser.add_argument("--subset", choices=get_args(_SUBSET_TYPE), required=True)
    parser.add_argument("--data_direc", type=str, default="../data/")
    # parser.add_argument('--save', type=bool, default=True)  # Not used
    parser.add_argument("--AD_Name", type=str, required=True)
    args = parser.parse_args()

    file_names = [
        f"TSB-AD-U/{file_name}"
        for file_name in pd.read_csv(
            f"{args.data_direc}/metadata/TSB-AD-U-{args.subset}.csv"
        )["file_name"]
    ]

    print("Model:", args.AD_Name)
    print("Number time series:", len(file_names))

    results = []
    for file_name in tqdm.tqdm(file_names):
        df = pd.read_csv(args.data_direc + file_name).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df["Label"].astype(int).to_numpy()

        slidingWindow = find_length_rank(data, rank=1)
        train_index = file_name.split(".")[0].split("_")[-3]
        data_train = data[: int(train_index), :]
        Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]

        start_time = time.time()
        if args.AD_Name in Semisupervise_AD_Pool:
            output = run_Semisupervise_AD(
                args.AD_Name, data_train, data, **Optimal_Det_HP
            )
        elif args.AD_Name in Unsupervise_AD_Pool:
            output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
        else:
            raise Exception(f"{args.AD_Name} is not defined")
        total_time = time.time() - start_time

        if isinstance(output, np.ndarray):
            output = (
                MinMaxScaler(feature_range=(0, 1))
                .fit_transform(output.reshape(-1, 1))
                .ravel()
            )
            evaluation_result = get_metrics(
                output,
                label,
                slidingWindow=slidingWindow,
                pred=output > (np.mean(output) + 3 * np.std(output)),
            )
            # print('Evaluation Result: ', evaluation_result)
            evaluation_result["Runtime [s]"] = total_time
            results.append(evaluation_result)

        else:
            print(f"At {file_name}: " + output)

    # Save overall results
    df = pd.DataFrame(results)

    base_path = pathlib.Path("results/raw/tsb_ad_reproduced").resolve()
    if not base_path.exists():
        base_path.mkdir(parents=True)

    # Path to the results file
    results_path = (
        base_path
        / f"TSB-AD-U-{args.subset}_{args.AD_Name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )

    # Saving the results
    df.to_csv(results_path, index=False)
