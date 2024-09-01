import os.path
import sys
sys.path.append('./attention')
from attention.Dataset.DatasetBC import load_dataset_custom
import torch
import numpy as np
from attention.attack import PGDAttacker
import time
if args.train_mode == "adv_train":
    print(args.train_mode)
    basedir = args.output_dir
    from attention.utlis.common import get_latest_model
    exp_name_load = '+'.join(('std_train',args.encoder, args.attention))
    args.gold_label_dir = get_latest_model(os.path.join(basedir,args.dataset,exp_name_load))
import wandb
wandb.init(project="SEAT",config=args)
wandb.log(vars(args))
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.train_mode == "adv_train":
    args.frozen_attn = False
    args.pre_loaded_attn = False
elif args.attention == 'tanh':
    args.frozen_attn = False
    args.pre_loaded_attn = False
else:
    raise LookupError("Attention not found")
dataset = load_dataset_custom(dataset_name=args.dataset, args=args)
from attention.configurations import generate_config
from attention.Trainers.TrainerBC import Trainer, Evaluator
config = generate_config(dataset, args, exp_name)
trainer = Trainer(dataset, args, config=config)
dirname = trainer.model.save_values(save_model=False)
print("DIRECTORY:", dirname)
PGDer = PGDAttacker(
        radius=args.pgd_radius, steps=args.pgd_step, step_size=args.pgd_step_size, random_start= \
        True, norm_type=args.pgd_norm_type, ascending=True
    )
X_PGDer = PGDAttacker(
    radius=args.x_pgd_radius, steps=args.x_pgd_step, step_size=args.x_pgd_step_size, random_start= \
        True, norm_type=args.x_pgd_norm_type, ascending=True
)
if args.train_mode == "std_train":
    trainer.train_standard(dataset, args, save_on_metric=args.save_on_metric)
elif args.train_mode == "adv_train":    
    trainer.PGDer = PGDer
    trainer.X_PGDer = X_PGDer
    trainer.train_ours(dataset, args)
evaluator = Evaluator(dirname, args)
comp, suff = evaluator.model.eval_comp_suff(dataset['test'], X_PGDer=X_PGDer, args=args)
sens = evaluator.model.eval_sens(dataset['test'], X_PGDer=X_PGDer, args=args)
wandb.log({
"suff_te": suff,
"comp_te": comp,
"sens_te": sens,
})
final_metric,_,_ = evaluator.evaluate(dataset['test'], save_results=True)
if args.train_mode == "adv_train":
    evaluator.model.end_clean()
wandb.log({
    "final_metric":final_metric
})
wandb.finish()
import os
import signal
os.kill(os.getpid(), signal.SIGKILL)
