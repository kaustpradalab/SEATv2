import wandb
import pickle
wandb.init(project="SEAT",mode='disabled')
runs=wandb.Api().runs("SEAT")
for run in runs:
    summary=run.summary_metrics
    epoch=summary.get('n_epoch')
    if epoch==40:
        topk_overlap_tr = run.history(keys=["topk_overlap_tr"])
        with open('./2/percent_tr_'+summary.get('dataset')+'.pkl', 'wb') as file:
            pickle.dump(topk_overlap_tr, file)
        topk_overlap_te = run.history(keys=["topk_overlap_te"])
        with open('./2/percent_te_'+summary.get('dataset')+'.pkl', 'wb') as file:
            pickle.dump(topk_overlap_te, file)
        topk_overlap_tr = run.history(keys=["sim_topk_loss_tr"])
        with open('./2/loss_tr_'+summary.get('dataset')+'.pkl', 'wb') as file:
            pickle.dump(topk_overlap_tr, file)
        topk_overlap_te = run.history(keys=["sim_topk_loss_te"])
        with open('./2/loss_te_'+summary.get('dataset')+'.pkl', 'wb') as file:
            pickle.dump(topk_overlap_te, file)