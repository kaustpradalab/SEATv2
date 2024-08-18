import wandb
import pandas as pd

def flatten_dict_to_list(d, parent_keys=[]):
    items = []
    for key, value in d.items():
        current_keys = parent_keys + [key]
        if isinstance(value, dict):
            items.extend(flatten_dict_to_list(value, current_keys))
        else:
            items.append(current_keys + value)
    return items

def add_data_to_nested_dict(data_dict, keys, value):
    current_dict = data_dict
    for key in keys[:-1]:
        if key not in current_dict:
            current_dict[key] = {}
        current_dict = current_dict[key]

    current_dict[keys[-1]] = value

wandb.init(project="SEAT",mode="disabled")
runs = wandb.Api().runs("SEAT")
data = {
}
flag=0
for run in runs:

    summary = run.summary_metrics
    encoder = summary.get("encoder")
    method = summary.get("method")
    if encoder == None:
        continue
    if method != "ours":
        continue
    comp_te = summary.get("comp_te")  
    suff_te = summary.get("suff_te")
    sens_te = summary.get("sens_te")
    
    jsd = summary["px_jsd_att_diff_te"]
    tvd = summary["px_tvd_pred_diff_te"]
    f1_score = summary["test_metrics"]["weighted avg/f1-score"]
    dataset = summary.get("dataset")
    x_rad = str(summary.get("x_pgd_radius"))
    base_comp_te = summary.get("baseline_comp_te")  
    base_suff_te = summary.get("baseline_suff_te")
    base_sens_te = summary.get("baseline_sens_te")
    base_jsd= summary["baseline_px_jsd_att_diff_te"]
    base_tvd= summary["baseline_px_tvd_pred_diff_te"]
    add_data_to_nested_dict(data, [x_rad, encoder,dataset, method], [comp_te,suff_te,sens_te,jsd,tvd,f1_score])
    add_data_to_nested_dict(data, [x_rad, encoder, dataset, "vanilla"], [base_comp_te,\
                                        base_suff_te,base_sens_te,base_jsd,base_tvd,f1_score])

import csv
header = ["x_rad","encoder","method","dataset","comp","suff","sens","jsd","tvd","f1-score"]
flat_dict =flatten_dict_to_list(data)
print(flat_dict[1:5])
 
with open('test.csv', mode='w', newline='',encoding='utf-8') as f:
    writer = csv.writer(f) 
    writer.writerow(header)  
    writer.writerows(flat_dict)  

