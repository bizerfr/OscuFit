import os, sys
import argparse
import torch
import torch.utils.data
import torch.utils.tensorboard
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch.nn.parallel

from utils.misc import *
#from network.model import DGCNN
from network.model import GraphCNN
from network.quadric import QuadraticSurface
from dataset import PointCloudDataset, PatchDataset, RandomPointcloudPatchSampler


def parse_arguments():
    parser = argparse.ArgumentParser()
    ### Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_gamma', type=float, default=0.2)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--use_normal', type=eval, default=True, choices=[True, False])
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--nepoch', type=int, default=500)
    parser.add_argument('--scheduler_epoch', type=int, nargs='+', default=[200,400,600,800])
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
    ### Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='/mnt/disk1/Dataset/')
    parser.add_argument('--data_set', type=str, default='ABC-Diff', choices=['ABC-Diff'])
    parser.add_argument('--trainset_list', type=str, default='trainingset_all.txt')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='The number of patches sampled from each shape in an epoch')
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    def worker_init_fn(worker_id):
        random.seed(args.seed)
        np.random.seed(args.seed)

    train_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='train',
            data_set=args.data_set,
            data_list=args.trainset_list,
        )
    train_set = PatchDataset(
            datasets=train_dset,
            patch_size=args.patch_size,
        )
    train_datasampler = RandomPointcloudPatchSampler(train_set, patches_per_shape=args.patches_per_shape, seed=args.seed)
    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            sampler=train_datasampler,
            batch_size=args.batch_size,
            num_workers=int(args.num_workers),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            # generator=g,
        )

    return train_dataloader, train_datasampler


### Arguments
args = parse_arguments()
seed_all(args.seed)

assert args.gpu >= 0, "ERROR GPU ID!"
device = torch.device('cuda:%d' % args.gpu)
PID = os.getpid()

### Datasets and loaders
print('Loading datasets ...')
train_dataloader, train_datasampler = get_data_loaders(args)
train_num_batch = len(train_dataloader)

### Model
print('Building model ...')
model = GraphCNN(num_points=args.patch_size, use_feat_stn=True)

model = model.to(device)
qsurf = QuadraticSurface().to(device)

### Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)


### Logging
if args.logging:
    log_path, log_dir_name = get_new_log_dir(args.log_root, prefix='',
                                            postfix='_' + args.tag if args.tag is not None else '')
    sub_log_dir = os.path.join(log_path, 'log')
    os.makedirs(sub_log_dir)
    logger = get_logger(name='train(%d)(%s)' % (PID, log_dir_name), log_dir=sub_log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(sub_log_dir)
    log_hyperparams(writer, sub_log_dir, args)
    ckpt_mgr = CheckpointManager(os.path.join(log_path, 'ckpts'))
   
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()

refine_epoch = -1
if args.resume != '':
    assert os.path.exists(args.resume), 'ERROR path: %s' % args.resume
    logger.info('Resume from: %s' % args.resume)

    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['state_dict'])
    refine_epoch = ckpt['others']['epoch']

    logger.info('Load pretrained mode: %s' % args.resume)

if args.logging:
    code_dir = os.path.join(log_path, 'code')
    os.makedirs(code_dir, exist_ok=True)
    os.system('cp %s %s' % ('*.py', code_dir))
    os.system('cp -r %s %s' % ('network', code_dir))
    os.system('cp -r %s %s' % ('utils', code_dir))


### Arguments
logger.info('Command: {}'.format(' '.join(sys.argv)))
arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
logger.info('Arguments:\n' + arg_str)
logger.info(repr(model))
logger.info('training set: %d patches (in %d batches)' %
                (len(train_datasampler), len(train_dataloader)))

torch.manual_seed(2025)

first_exploded = {"name": None, "norm": 0.0, "printed": False}

def register_grad_hooks(model, threshold=100.0):
    def grad_hook(name):
        def hook(grad):
            norm = grad.norm().item()
            if not torch.isnan(grad).any() and norm > threshold:
                if not first_exploded["printed"]:
                    first_exploded["name"] = name
                    first_exploded["norm"] = norm
                    first_exploded["printed"] = True
                    print(f"💥 GLOBAL FIRST explosion: {name} | grad norm = {norm:.2f}")
                    import pdb; pdb.set_trace()
            return grad
        return hook

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(grad_hook(name))




def train(epoch):
    Loss_items = None
    Loss = 0.0
    for train_batchind, batch in enumerate(train_dataloader, 0):
        points_patch = batch['points_patch'].to(device)
        center_normals = batch['center_normals'].to(device)  
        center_curvatures = batch['center_curvatures'].to(device)
        center_directions = batch['center_directions'].to(device)

        center_kg = center_curvatures[:, 0] * center_curvatures[:, 1]
        center_km = 0.5 * (center_curvatures[:, 0] + center_curvatures[:, 1])
       
        ### Reset grad and model state
        model.train()
        
        #register_grad_hooks(model, threshold=100)
        optimizer.zero_grad()
        

        ### Forward

        weights, offsets = model(points_patch.transpose(2, 1))
        u = qsurf.fit(weights, points_patch + offsets.transpose(2, 1))
        
        n_est = qsurf.get_origin_normal(u)
        kg_est, km_est = qsurf.get_origin_gaussian_mean_curvatures(u)
        u_gt = qsurf.get_transformed_monge(center_curvatures, center_directions, center_normals)

        loss, loss_dict = model.get_loss(n_est, center_normals, 
                                         kg_est, km_est, center_kg, center_km,
                                         u, u_gt)

        if Loss_items is None:
            Loss_items = {key: 0.0 for key in loss_dict.keys()}
 
        writer.add_scalar('Training loss - iterations', loss, epoch * (train_num_batch-1) + train_batchind)
        for key, value in loss_dict.items():
            writer.add_scalar('Training ' + key + ' iterations', value.item(), epoch * (train_num_batch-1) + train_batchind)
            Loss_items[key] += value
        
        Loss += loss
        
        ### Backward and optimize
        loss.backward()

        grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        ### Logging
        loss_components = ",".join([f"{value.item():.6f}" for key, value in loss_dict.items()])
        logger.info(
            f'[Train] [{epoch:03d}: {train_batchind:03d}/{train_num_batch - 1:03d}] | '
            f'loss: {loss.item():.6f} ({loss_components}) | Grad: {grad_norm:.6f}'
            )

    writer.add_scalar('Training Loss - epoch', Loss, epoch)
    for key, total_value in Loss_items.items():
        writer.add_scalar('Training '+ key + ' epoch', total_value, epoch)

def scheduler_fun():
    pre_lr = optimizer.param_groups[0]['lr']
    current_lr = pre_lr * args.lr_gamma
    if current_lr < args.lr_min:
        current_lr = args.lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    logger.info('Update learning rate: %f => %f \n' % (pre_lr, current_lr))


if __name__ == '__main__':
    logger.info('Start training ...')
    try:
        for epoch in range(1, args.nepoch+1):
            logger.info('### Epoch %d ###' % epoch)
            if epoch <= refine_epoch:
                # scheduler.step()
                if epoch in args.scheduler_epoch:
                    scheduler_fun()
                continue

            start_time = time.time()
            train(epoch)
            end_time = time.time()
            logger.info('Time cost: %.1f s \n' % (end_time-start_time))

            # scheduler.step()
            if epoch in args.scheduler_epoch:
                scheduler_fun()

            if epoch % args.interval == 0 or epoch == args.nepoch-1:
                opt_states = {
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict(),
                }

                if args.logging:
                    ckpt_mgr.save(model, args, others=opt_states, step=epoch)

    except KeyboardInterrupt:
        logger.info('Terminating ...')
