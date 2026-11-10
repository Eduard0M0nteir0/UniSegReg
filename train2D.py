"""UniSegReg 2D training on the PAS testis dataset built by prepare_data_2d.py.

Differences from the published train2D.py, all marked `# CHANGED` below:
  1. data_path points at ./data_2d and every setting is overridable from the CLI.
  2. The labeled set is selected by file-name prefix (`src_` vs `unl_`) instead of
     `source = train[:num_source]`, which slices a shuffled list and would mix
     unlabeled patches into the supervised set.
  3. cfg.pre no longer applies OneHot([0, 255]) to the labels: lab_b.npy already
     holds a one-hot [2, H, W] float32 map. That is the only layout that works for
     both training and the quality-assessment loop (`ass_pre` does not one-hot, and
     QualityAssessment writes its soft pseudo-labels in exactly this layout).
  4. pre_train_reg is on. The original restored the registration net from a
     checkpoint next to the output folder; no such checkpoint exists here.
  5. The unlabeled target pool and the registration pre-training pair list can be
     capped (--max-targets / --max-reg-pairs), and the schedule defaults are sized
     for a CPU run instead of the paper's.
  6. Missing checkpoints raise an explanatory error instead of a bare traceback.
  7. Augmentation is on for segmentation training (`--aug dihedral`): exact flips and
     90-degree rotations applied to image, label and weight map together, plus gamma and
     noise jitter. See A.RandomDihedral2D for why A.RandomAugmentation is not used.
  8. `--loss` selects the segmentation loss. The default `dice_ce_boundary` scores Dice
     over both classes and weights cross entropy near label boundaries, which is aimed
     at the failure seen in the first run (precision 0.47 at recall 0.96: predictions
     bled across the septa and merged neighbouring tubules). `--loss dice_ce` restores
     the published setting. cfg.val_loss_key / save_frequency / early_stop_patience are
     now actually passed to the trainer.

Run `python train2D.py --help` for the options, `--quick` for a smoke test.
"""

import argparse
import glob
import os
import shutil

import numpy as np
import scipy.io as sio

import torch
from torch.utils.data.dataloader import DataLoader

from snmi.core import Trainer
from snmi._custom import SegRegDataset
from snmi.utils import augmentation_methods as A, loss_fuctions as LF, process_methods as P, utils as U

from snmi.nets.net_reg import MrReg
from snmi.nets.unet2D import Unet2D
from snmi._custom.quality_assessment_float import QualityAssessment
from models import JointSegModel, JointRegModel

HERE = os.path.dirname(os.path.abspath(__file__))


# CHANGED: CLI so the dataset path and the (expensive) schedule are not hard-coded.
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-path', default=os.path.join(HERE, 'data_2d'),
                    help='dataset built by prepare_data_2d.py')
    ap.add_argument('--output-path', default=os.path.join(HERE, 'results_2D'))
    ap.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    ap.add_argument('--label-format', choices=('onehot', 'binary255'), default='onehot',
                    help='must match prepare_data_2d.py --label-format')
    ap.add_argument('--quick', action='store_true',
                    help='tiny schedule to check the pipeline end to end (minutes)')
    ap.add_argument('--test-only', action='store_true',
                    help='skip all training, only test from existing checkpoints')
    ap.add_argument('--iters', type=int, default=3, help='joint seg/reg iterations (paper: 10)')
    ap.add_argument('--pre-seg-ep', type=int, default=400, help='paper: 500')
    ap.add_argument('--pre-reg-ep', type=int, default=40, help='paper: 200')
    ap.add_argument('--seg-ep', type=int, default=30, help='paper: 100')
    ap.add_argument('--reg-ep', type=int, default=10, help='paper: 50')
    ap.add_argument('--loss', choices=('dice_ce', 'dice_ce_bg', 'dice_ce_boundary'),
                    default='dice_ce_boundary',
                    help='segmentation loss: dice_ce = paper (foreground-only Dice + CE); '
                         'dice_ce_bg = Dice over both classes; dice_ce_boundary = adds the '
                         'boundary-weighted CE that penalizes bleeding across septa')
    ap.add_argument('--boundary-w0', type=float, default=10.0, help='boundary weight height')
    ap.add_argument('--boundary-sigma', type=float, default=5.0,
                    help='boundary weight width in pixels (~ the gaps to keep open)')
    ap.add_argument('--aug', choices=('none', 'dihedral'), default='dihedral',
                    help='augmentation for segmentation training')
    ap.add_argument('--aug-rate', type=float, default=0.75,
                    help='probability of a flip/rotation per sample')
    ap.add_argument('--no-jitter', action='store_true',
                    help='disable the gamma/noise jitter that comes with --aug dihedral')
    ap.add_argument('--early-stop-patience', type=int, default=0,
                    help='stop pre-training after N evaluations without improvement (0 = off)')
    ap.add_argument('--batch-size', type=int, default=5)
    ap.add_argument('--max-targets', type=int, default=400,
                    help='cap on unlabeled targets kept for pseudo-labeling (0 = all)')
    ap.add_argument('--max-reg-pairs', type=int, default=600,
                    help='cap on image pairs used for registration pre-training (0 = all)')
    ap.add_argument('--num-assessment', type=int, default=5,
                    help='atlases/augmentations averaged per target in the assessment')
    ap.add_argument('--log-images', action='store_true',
                    help='write example images to tensorboard (slow: matplotlib per batch)')
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args(argv)
    if args.quick:
        args.pre_seg_ep, args.pre_reg_ep = 2, 2
        args.seg_ep, args.reg_ep, args.iters = 2, 2, 1
        args.max_targets, args.max_reg_pairs, args.num_assessment = 20, 40, 2
    return args


args = parse_args()


# CHANGED: selectable segmentation loss (item 1 of the post-run diagnosis).
#   dice_ce          what the paper used: SoftDice ignores the background channel and CE
#                    hardens the target with argmax.
#   dice_ce_bg       Dice over both channels, so background errors cost something.
#   dice_ce_boundary adds BoundarySeparationCrossEntropy, which weights pixels near a
#                    label boundary and keeps soft pseudo-labels soft.
def build_seg_loss(name):
    if name == 'dice_ce':
        return {LF.SoftDice(): 0.5, LF.CrossEntropy(): 0.5}
    if name == 'dice_ce_bg':
        return {LF.SoftDice(ignore_background=False): 0.5, LF.CrossEntropy(): 0.5}
    return {LF.SoftDice(ignore_background=False): 0.5,
            LF.BoundarySeparationCrossEntropy(w0=args.boundary_w0,
                                              sigma=args.boundary_sigma): 0.5}


def pick_device(choice):
    if choice == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda requested but torch.cuda.is_available() is False '
                         f'(torch {torch.__version__}).')
    if choice == 'cpu' or (choice == 'auto' and not torch.cuda.is_available()):
        return torch.device('cpu')
    return torch.device('cuda')


def require_ckpt(path, hint):
    if not os.path.isfile(path):
        raise SystemExit(f'Checkpoint not found: {path}\n  {hint}')
    return torch.load(path, map_location='cpu')


class cfg:
    data_path = args.data_path          # CHANGED: prepared PAS testis patches
    output_path = args.output_path

    test_only = args.test_only          # CHANGED: train by default
    test_pre_seg = True
    test_pre_reg = True                 # CHANGED: source_test/ exists, so reg can be tested
    test_seg = True
    test_reg = True                     # CHANGED

    # pre_training parameters
    pre_train_seg = True
    pre_train_reg = True                # CHANGED: no external pre_reg checkpoint exists
    pre_seg_ep = args.pre_seg_ep
    pre_seg_lr = 1e-4
    pre_seg_eval_frequency = max(1, args.pre_seg_ep // 20)   # CHANGED: scales with the schedule
    pre_seg_loss = build_seg_loss(args.loss)                 # CHANGED: --loss

    pre_reg_ep = args.pre_reg_ep
    pre_reg_lr = 1e-3
    pre_reg_eval_frequency = max(1, args.pre_reg_ep // 5)    # CHANGED
    pre_reg_loss = {LF.GNCC(): 1}
    pre_disp_loss = {LF.Grad('l2'): [128, 64, 32, 16, 8]}

    # training parameters
    train_batch_size = args.batch_size
    eval_batch_size = args.batch_size
    eval_frequency = max(1, min(args.seg_ep, args.reg_ep) // 2)   # CHANGED

    # iterative
    iters_start = 0
    iters = args.iters
    threshold_dice = 0.9
    threshold_dice_min = 0.2
    num_assessment = args.num_assessment
    # min_good is set below from the actual target count.
    # NOTE: snmi/_custom/quality_assessment_float.py hard-codes the per-target score to
    # 1 (`dice = [1]`), so every target passes threshold_dice and min_good/thresholds
    # have no effect. Selection is disabled in this float variant; confidence is
    # carried by the per-pixel weight map instead.

    # segmentation setting
    seg_ep = args.seg_ep
    seg_lr = 1e-5
    seg_loss = build_seg_loss(args.loss)    # CHANGED: --loss
    seg_layers = 5
    seg_dropout_rate = 0.2

    # registration setting
    reg_ep = args.reg_ep
    reg_lr = 1e-4
    freeze_reg = False

    reg_loss = {LF.GNCC(): [1, 1, 1, 1, 1]}
    disp_loss = {LF.Grad('l2'): [128, 64, 32, 16, 8]}
    lab_loss = {LF.RegDice(weight=False): 1}
    reg_layers = 5
    int_steps = 0

    # training setttings
    log_train_image = args.log_images
    seg_log_validation_image = args.log_images
    reg_log_validation_image = args.log_images   # CHANGED: off by default (slow on CPU)
    save_frequency = None
    val_loss_key = 'loss'   # key of evaluation result to control early stop and best ckpt save
    early_stop_patience = args.early_stop_patience or None   # CHANGED: now actually passed
    learning_rate_schedule = None

    # set the suffix for data loading
    img_suffix = 'org.tif'
    lab_suffix = 'lab_b.npy'
    weight_suffix = 'weight.tif'
    source_key = 'source'
    target_key = 'target'

    # CHANGED: lab_b.npy is already one-hot [2,H,W] float32 -> no OneHot here.
    pre = {img_suffix: [P.min_max, P.ExpandDim(0)],
           lab_suffix: [] if args.label_format == 'onehot' else [P.OneHot([0, 255])],
           weight_suffix: [lambda img: np.array(img > 0.5, np.float32), P.ExpandDim(0)]}


# Data ------------------------------------------------------------------------
def list_split(name):
    files = sorted(glob.glob(os.path.join(cfg.data_path, name, '*' + cfg.img_suffix)))
    if not files:
        raise SystemExit(f'No "*{cfg.img_suffix}" in {os.path.join(cfg.data_path, name)}.\n'
                         f'  Build the dataset first: python prepare_data_2d.py')
    return files


def subsample_by_case(files, limit, rng):
    """Keep `limit` files, round-robin over source images so every case stays in."""
    if limit <= 0 or limit >= len(files):
        return files
    groups = {}
    for f in files:
        case = os.path.basename(f).split('_x')[0]
        groups.setdefault(case, []).append(f)
    for case in groups:
        rng.shuffle(groups[case])
    order = sorted(groups)
    kept, i = [], 0
    while len(kept) < limit:
        progressed = False
        for case in order:
            if i < len(groups[case]):
                kept.append(groups[case][i])
                progressed = True
                if len(kept) == limit:
                    break
        if not progressed:
            break
        i += 1
    return sorted(kept)


rng = np.random.RandomState(args.seed)

train = list_split('train')
valid = list_split('valid')
test = list_split('test')
source_test = list_split('source_test')

# CHANGED: labeled vs. unlabeled by prefix instead of slicing a shuffled list.
source = [f for f in train if os.path.basename(f).startswith('src_')]
target_train = [f for f in train if os.path.basename(f).startswith('unl_')]
if not source:
    raise SystemExit(f'No labeled "src_*" patches in {cfg.data_path}/train.')
target_train = subsample_by_case(target_train, args.max_targets, rng)

target_valid = valid
target_test = test
cfg.min_good = max(1, len(target_train) // 2)

print(f'Dataset   {cfg.data_path}')
print(f'  train   {len(source)} labeled source + {len(target_train)} unlabeled target patches'
      + (f' (capped from {len(train) - len(source)})' if args.max_targets else ''))
print(f'  valid   {len(target_valid)}   test {len(target_test)}   '
      f'source_test {len(source_test)}')
print(f'Training  loss={args.loss}  aug={args.aug}'
      + (f' (rate {args.aug_rate}, jitter {not args.no_jitter})' if args.aug != 'none' else '')
      + f'  pre_seg_ep={cfg.pre_seg_ep} seg_ep={cfg.seg_ep} iters={cfg.iters}')

# source check
source_check_path = cfg.output_path + '/source_start'
if not os.path.exists(source_check_path):
    os.makedirs(source_check_path)
[shutil.copy2(x, source_check_path) for x in source]
[shutil.copy2(x.replace(cfg.img_suffix, cfg.lab_suffix), source_check_path) for x in source]

# Network
net_seg = Unet2D(n_classes=2, in_channels=1, n_layers=cfg.seg_layers)
net_reg = MrReg(n_layers=cfg.reg_layers, int_steps=cfg.int_steps)

# Optimizer
pre_opt_seg = torch.optim.Adam(net_seg.parameters(), lr=cfg.pre_seg_lr)
pre_opt_reg = torch.optim.Adam(net_reg.parameters(), lr=cfg.pre_reg_lr)
opt_seg = torch.optim.Adam(net_seg.parameters(), lr=cfg.seg_lr)
opt_reg = torch.optim.Adam(net_reg.parameters(), lr=cfg.reg_lr)

# device
device = pick_device(args.device)   # CHANGED: --device, with an explicit error for cuda
print(f'Device    {device}'
      + ('' if device.type == 'cuda' else f' (torch {torch.__version__}, no CUDA)'))

# pre train
# CHANGED: augmentation is on. A.RandomAugmentation would rotate the channel axis of a
# one-hot [2,H,W] label and would leave the weight map unaugmented, so this uses
# RandomDihedral2D, which transforms image, label and weight map together on the last
# two axes with exact flips/90-degree rotations.
if args.aug == 'none':
    aug = None
else:
    aug = A.RandomDihedral2D(
        image_keys=[cfg.target_key + cfg.img_suffix],
        map_keys=[cfg.target_key + cfg.lab_suffix, cfg.target_key + cfg.weight_suffix],
        aug_rate=args.aug_rate,
        jitter=None if args.no_jitter else A.IntensityJitter())
seg_train_set = SegRegDataset([], source, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre, aug)
seg_train_loader = DataLoader(seg_train_set, batch_size=cfg.train_batch_size, shuffle=True)

# CHANGED: registration pre-training pairs drawn from source + (capped) targets.
reg_pool = subsample_by_case(source + target_train, args.max_reg_pairs, rng)
reg_target_list = list(reg_pool)
rng.shuffle(reg_target_list)
reg_train_set = SegRegDataset(reg_pool, reg_target_list, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
reg_train_loader = DataLoader(reg_train_set, batch_size=cfg.train_batch_size, shuffle=True)

valid_set = SegRegDataset(source, target_valid, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
valid_loader = DataLoader(valid_set, batch_size=cfg.eval_batch_size)

pre_model_seg = JointSegModel(net_seg, pre_opt_seg, cfg.img_suffix, cfg.lab_suffix, cfg.target_key, weight_suffix=cfg.weight_suffix,
                     dropout_rate=cfg.seg_dropout_rate, loss_functions=cfg.pre_seg_loss, device=device)
pre_model_reg = JointRegModel(net_reg, pre_opt_reg, cfg.img_suffix, cfg.source_key, cfg.target_key, lab_suffix=cfg.lab_suffix, weight_suffix=cfg.weight_suffix,
                    loss_functions=cfg.reg_loss, disp_loss_functions=cfg.disp_loss, lab_loss_functions=None, device=device)

pre_trainer_seg = Trainer(pre_model_seg)
pre_trainer_reg = Trainer(pre_model_reg)

if not cfg.test_only :
    if cfg.pre_train_seg:
        pre_trainer_seg.train(seg_train_loader, valid_loader, 
            epochs=cfg.pre_seg_ep, 
            output_path=cfg.output_path + '/pre_seg', 
            log_train_image=cfg.log_train_image,   # CHANGED: was hard-coded True
            log_validation_image=cfg.seg_log_validation_image,
            eval_frequency=cfg.pre_seg_eval_frequency,
            val_loss_key=cfg.val_loss_key,                       # CHANGED: cfg was unused
            early_stop_patience=cfg.early_stop_patience,         # CHANGED
            save_frequency=cfg.save_frequency,                   # CHANGED
            print_tag=f'pre Seg - ')
    if cfg.pre_train_reg:
        pre_trainer_reg.cur_epoch = 0
        pre_trainer_reg.train(reg_train_loader, valid_loader, 
            epochs=cfg.pre_reg_ep, 
            output_path=cfg.output_path + '/pre_reg', 
            log_train_image=cfg.log_train_image, 
            log_validation_image=cfg.reg_log_validation_image,
            eval_frequency=cfg.pre_reg_eval_frequency,
            print_tag=f'pre Reg - ')

# Joint Model and Trainer
model_seg = JointSegModel(net_seg, opt_seg, cfg.img_suffix, cfg.lab_suffix, cfg.target_key, weight_suffix=cfg.weight_suffix,
                     dropout_rate=cfg.seg_dropout_rate, loss_functions=cfg.seg_loss, device=device)
model_reg = JointRegModel(net_reg, opt_reg, cfg.img_suffix, cfg.source_key, cfg.target_key, lab_suffix=cfg.lab_suffix, weight_suffix=cfg.weight_suffix,
                    loss_functions=cfg.reg_loss, disp_loss_functions=cfg.disp_loss, lab_loss_functions=cfg.lab_loss, device=device)

trainer_seg = Trainer(model_seg)
trainer_reg = Trainer(model_reg)

# Assessment initialization
count_good = 0
assessment = QualityAssessment(model_seg, model_reg)
ass_pre = {cfg.img_suffix: [P.min_max, P.ExpandDim(0), P.ToTensor(0)],
            cfg.lab_suffix: [P.ToTensor(0)]}
target_set = SegRegDataset(source, target_train, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix], ass_pre, is_assessment=True, num_assessment=cfg.num_assessment)
if cfg.iters_start == 0:
    # Restore pre-trained model   # CHANGED: both branches now read this run's output
    ckpt_seg = require_ckpt(cfg.output_path + '/pre_seg/ckpt/model_best.pt',
                            'Run without --test-only first to pre-train the segmentation net.')
    ckpt_reg = require_ckpt(cfg.output_path + '/pre_reg/ckpt/model_final.pt',
                            'Run without --test-only first to pre-train the registration net.')
    net_seg.load_state_dict(ckpt_seg['net0'])
    net_reg.load_state_dict(ckpt_reg['net0'])
    # freeze encoder and first 3 decoder of registration model
    if cfg.freeze_reg:
        net_reg.enc.requires_grad_(False)
        for name, param in net_reg.named_parameters():
            print(name,param.requires_grad)

    # pre test
    print('pre test')
    with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
            f.write(f'Baseline: \n')
    if cfg.pre_train_seg or cfg.test_pre_seg:
        seg_test_set = SegRegDataset([], target_test, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
        seg_test_loader = DataLoader(seg_test_set, batch_size=cfg.eval_batch_size)
        seg_results = trainer_seg.test(seg_test_loader, cfg.output_path + '/pre_seg', desc=f'Pre Test Seg: ', log_image=cfg.seg_log_validation_image)
        sio.savemat(f'{cfg.output_path}/pre_seg/test_results.mat', seg_results)
        with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
            f.write(f'  Seg - {U.dict_to_str(seg_results)}\n')
    if cfg.pre_train_reg or cfg.test_pre_reg:
        reg_test_set = SegRegDataset(sorted(source_test), sorted(target_test*len(source_test)), cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
        reg_test_loader = DataLoader(reg_test_set, batch_size=len(source_test))
        reg_results = trainer_reg.test(reg_test_loader, cfg.output_path + '/pre_reg', desc=f'Pre Test Reg: ', log_image=cfg.reg_log_validation_image)
        sio.savemat(f'{cfg.output_path}/pre_reg/test_results.mat', reg_results)
        with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
            f.write(f'  Reg - {U.dict_to_str(reg_results)}\n')
    seg_results = None 
    reg_results = None

    # generate assessed data
    if not cfg.test_only:
        count_good = assessment.assessment(target_set, cfg.output_path+'/assessed_data', 
                                cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix,
                                [0,1], cfg.threshold_dice, cfg.threshold_dice_min, cfg.min_good, target_key=cfg.target_key)
        with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
            f.write(f'  count good - {count_good}/{len(target_train)}\n\n')
        count_good = 0

# Iterative training
for it_idx in range(cfg.iters_start, cfg.iters):
    # Path
    it_output_path = f'{cfg.output_path}/iter-{it_idx}'
    it_path_seg = f'{it_output_path}/seg'
    it_path_reg = f'{it_output_path}/reg'

    # iter restore
    if it_idx == 0:
        it_target_list = glob.glob(f'{cfg.output_path}/assessed_data/good/*{cfg.img_suffix}')
    else:
        it_target_list = glob.glob(f'{cfg.output_path}/iter-{it_idx-1}/assessed_data/good/*{cfg.img_suffix}')
        trainer_seg.restore(f'{cfg.output_path}/iter-{it_idx-1}/seg/ckpt/model_best.pt')
        trainer_reg.restore(f'{cfg.output_path}/iter-{it_idx-1}/reg/ckpt/model_final.pt')

    # CHANGED: explain an empty pool instead of failing inside BaseDataset.
    if not cfg.test_only and not it_target_list:
        raise SystemExit(f'No pseudo-labeled targets for iteration {it_idx}. '
                         f'Expected files in {it_output_path}/../assessed_data/good.')

    # Train data loader
    seg_train_list = source + it_target_list
    seg_train_set = SegRegDataset([], seg_train_list, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre, aug)   # CHANGED: augment here too
    seg_train_loader = DataLoader(seg_train_set, batch_size=cfg.train_batch_size, shuffle=True)
    
    reg_train_list = source + it_target_list
    reg_target_list = reg_train_list.copy()
    np.random.shuffle(reg_target_list)
    reg_train_set = SegRegDataset(reg_train_list, reg_target_list, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
    reg_train_loader = DataLoader(reg_train_set, batch_size=cfg.train_batch_size, shuffle=True)

    valid_set = SegRegDataset(source, target_valid, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
    valid_loader = DataLoader(valid_set, batch_size=cfg.eval_batch_size)

    if not cfg.test_only:
        print(f'iter {it_idx}: {len(source)} labeled + {len(it_target_list)} pseudo-labeled patches')
        trainer_seg.cur_epoch = 0
        trainer_seg.train(seg_train_loader, valid_loader, 
            epochs=cfg.seg_ep, 
            output_path=it_path_seg, 
            log_train_image=cfg.log_train_image, 
            log_validation_image=cfg.seg_log_validation_image,
            eval_frequency=cfg.eval_frequency,
            val_loss_key=cfg.val_loss_key,                       # CHANGED
            save_frequency=cfg.save_frequency,                   # CHANGED
            print_tag=f'iter {it_idx} Seg - ')
        
        trainer_reg.cur_epoch = 0
        trainer_reg.train(reg_train_loader, valid_loader, 
            epochs=cfg.reg_ep, 
            output_path=it_path_reg, 
            log_train_image=cfg.log_train_image, 
            log_validation_image=cfg.reg_log_validation_image,
            eval_frequency=cfg.eval_frequency,
            print_tag=f'iter {it_idx} Reg - ')

        # assessment
        count_good = assessment.assessment(target_set, it_output_path+'/assessed_data', 
                          cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix,
                          [0,1], cfg.threshold_dice, cfg.threshold_dice_min, cfg.min_good * (it_idx+2), target_key=cfg.target_key)
            

    # test seg
    print()
    if cfg.test_seg:
        require_ckpt(f'{it_path_seg}/ckpt/model_best.pt',
                     f'Iteration {it_idx} was never trained; drop --test-only or lower --iters.')
        trainer_seg.restore(f'{it_path_seg}/ckpt/model_best.pt')
        seg_test_set = SegRegDataset([], target_test, cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
        seg_test_loader = DataLoader(seg_test_set, batch_size=cfg.eval_batch_size)
        it_seg_results = trainer_seg.test(seg_test_loader, it_path_seg, desc=f'iter {it_idx} Seg - Test: ', log_image=cfg.seg_log_validation_image)
        sio.savemat(f'{it_path_seg}/test_results.mat', it_seg_results)
        with open(f'{it_path_seg}/test_results.txt', 'a+') as f:
            f.write(U.dict_to_str(it_seg_results) + '\n')
    
    # test reg
    if cfg.test_reg:
        require_ckpt(f'{it_path_reg}/ckpt/model_best.pt',
                     f'Iteration {it_idx} was never trained; drop --test-only or lower --iters.')
        trainer_reg.restore(f'{it_path_reg}/ckpt/model_best.pt')
        reg_test_set = SegRegDataset(sorted(source_test), sorted(target_test*len(source_test)), cfg.source_key, cfg.target_key, [cfg.img_suffix, cfg.lab_suffix, cfg.weight_suffix], cfg.pre)
        reg_test_loader = DataLoader(reg_test_set, batch_size=len(source_test))
        it_reg_results = trainer_reg.test(reg_test_loader, it_path_reg, desc=f'iter {it_idx} Reg - Test: ', log_image=cfg.reg_log_validation_image)
        sio.savemat(f'{it_path_reg}/test_results.mat', it_reg_results)
        with open(f'{it_path_reg}/test_results.txt', 'a+') as f:
            f.write(U.dict_to_str(it_reg_results) + '\n')
    
    print()

    with open(f'{cfg.output_path}/final_test_results.txt', 'a+') as f:
            f.write(f'iter {it_idx}:\n')
            if cfg.test_seg:
                f.write(f'  Seg - {U.dict_to_str(it_seg_results)}\n')
            if cfg.test_reg:
                f.write(f'  Reg - {U.dict_to_str(it_reg_results)}\n')
            if count_good > 0:
                f.write(f'  count good - {count_good}/{len(target_train)}\n')
                count_good = 0
            f.write('\n')
    it_seg_results = None
    it_reg_results = None

print(f'Done. Results in {cfg.output_path} (summary: final_test_results.txt)')
