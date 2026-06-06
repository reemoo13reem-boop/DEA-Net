import os,argparse
import json


parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', type=str, default='/kaggle/working/experiment')
parser.add_argument('--dataset', type=str, default='NYU2')
parser.add_argument('--hazy_dir', type=str, default='/kaggle/input/datasets/reemsss/dea-net/training_images/data')
parser.add_argument('--clear_dir', type=str, default='/kaggle/input/datasets/reemsss/dea-net/original_image/image')
parser.add_argument('--model_name', type=str, default='DEA-Net', help='experiment name')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')

# only need for evaluation
parser.add_argument('--pre_trained_model', type=str, default='null', help='path of pre trained model for resume training')
parser.add_argument('--save_infer_results', action='store_true', default=False, help='save the infer results during validation')
opt=parser.parse_args()

exp_dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
exp_model_dir = os.path.join(exp_dataset_dir, opt.model_name)

if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)

if not os.path.exists(exp_dataset_dir):
    os.mkdir(exp_dataset_dir)

model_name = os.path.splitext(os.path.basename(opt.pre_trained_model))[0]
opt.saved_infer_dir = os.path.join(exp_model_dir, model_name)
if not os.path.exists(exp_model_dir):
    os.mkdir(exp_model_dir)
    os.mkdir(opt.saved_infer_dir)
if not os.path.exists(opt.saved_infer_dir):
    os.mkdir(opt.saved_infer_dir)

with open(os.path.join(exp_model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
