import os, sys
import shutil
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np

#from network.model import DGCNN
from network.model import GraphCNN
from network.quadric import QuadraticSurface
from utils.misc import get_logger, seed_all
from dataset import PointCloudDataset, PatchDataset, SequentialPointcloudPatchSampler, load_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default='/mnt/disk1/Dataset/')
    parser.add_argument('--data_set', type=str, default='ABC-Diff', choices=['ABC-Diff', 'pclouds', 'FamousShapes', 'SceneNN'])
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--ckpt_dirs', type=str, default='pretrain-v1', help="can be multiple directories, separated by ',' ")
    parser.add_argument('--ckpt_iter', type=str, default='500')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--testset_list', type=str, default='testset_all.txt')
    parser.add_argument('--eval_list', type=str,
                        default=['testset_no_noise.txt', 'testset_low_noise.txt', 'testset_med_noise.txt', 'testset_high_noise.txt',
                                'testset_vardensity_striped.txt', 'testset_vardensity_gradient.txt'],
                        nargs='*', help='list of .txt files containing sets of point cloud names for evaluation')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--sparse_patches', type=eval, default=True, choices=[True, False],
                        help='test on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--save_pn', type=eval, default=True, choices=[True, False])
    parser.add_argument('--save_pcurv', type=eval, default=True, choices=[True, False])
    parser.add_argument('--save_kg', type=eval, default=True, choices=[True, False])
    parser.add_argument('--save_km', type=eval, default=True, choices=[True, False])
    args = parser.parse_args()
    return args


def get_data_loaders(args):
    test_dset = PointCloudDataset(
            root=args.dataset_root,
            mode='test',
            data_set=args.data_set,
            data_list=args.testset_list,
            sparse_patches=args.sparse_patches,
        )
    test_set = PatchDataset(
            datasets=test_dset,
            patch_size=args.patch_size,
        )
    test_dataloader = torch.utils.data.DataLoader(
            test_set,
            sampler=SequentialPointcloudPatchSampler(test_set),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    return test_dset, test_dataloader


### Arguments
args = parse_arguments()
arg_str = '\n'.join(['    {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
print('Arguments:\n %s\n' % arg_str)

seed_all(args.seed)
PID = os.getpid()

assert args.gpu >= 0, "ERROR GPU ID!"
device = torch.device('cuda:%d' % args.gpu)

### Datasets and loaders
test_dset, test_dataloader = get_data_loaders(args)


def normal_RMSE(normal_gts, normal_preds, eval_file='log.txt'):
    """
        Compute normal root-mean-square error (RMSE)
    """
    def l2_norm(v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    log_file = open(eval_file, 'w')
    def log_string(out_str):
        log_file.write(out_str+'\n')
        log_file.flush()

    rms   = []

    for i in range(len(normal_gts)):
        normal_gt = normal_gts[i]
        normal_pred = normal_preds[i]

        normal_gt_norm = l2_norm(normal_gt)
        normal_results_norm = l2_norm(normal_pred)
        normal_pred = np.divide(normal_pred, np.tile(np.expand_dims(normal_results_norm, axis=1), [1, 3]))
        normal_gt = np.divide(normal_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3]))

        ### Unoriented rms
        nn = np.sum(np.multiply(normal_gt, normal_pred), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))

        ### Error metric
        rms.append(np.sqrt(np.mean(np.square(ang))))
        
    avg_rms  = np.mean(rms)


    log_string('RMS per shape: ' + str(rms))
    log_string('RMS not oriented (shape average): ' + str(avg_rms))

    log_file.close()

    return avg_rms

def map_pricipal_curvatures(current_curvatures):
    sorted_curvs = np.sort(np.abs(current_curvatures), axis=1)
    return sorted_curvs

def curvature_RMSE(pcurv_gts, pcurv_preds, kg_gts, kg_preds, km_gts, km_preds, eval_file='log.txt'):
    """
        Compute curvature root-mean-square error (RMSE)
    """
    log_file = open(eval_file, 'w')
    def log_string(out_str):
        log_file.write(out_str+'\n')
        log_file.flush()

    pcurv_rms = []
    pcurv_rectified_rms = []

    kg_rms = []
    kg_rectified_rms = []
    km_rms = []
    km_rectified_rms = []
   
    for i in range(len(pcurv_gts)):
        pcurv_gt = pcurv_gts[i]
        pcurv_pred = pcurv_preds[i]
        pcurv_pred = map_pricipal_curvatures(pcurv_pred)  
        pcurv_gt = map_pricipal_curvatures(pcurv_gt) 

        pcurv_rms_shape = np.sqrt(np.nanmean(np.square(pcurv_pred - pcurv_gt), axis=0))
        rectified_pcurv_rms_shape = np.sqrt(
            np.nanmean(
                np.square(
                    (np.abs(pcurv_pred) - np.abs(pcurv_gt)) / np.maximum(np.abs(pcurv_gt), np.ones_like(pcurv_gt))
                )
            , axis=0)
            )

        kg_gt = kg_gts[i]
        km_gt = km_gts[i]
        kg_pred = kg_preds[i]
        km_pred = km_preds[i]

        kg_rms_shape = np.sqrt(np.nanmean(np.square(kg_pred - kg_gt)))
        km_rms_shape = np.sqrt(np.nanmean(np.square(np.abs(km_pred) - np.abs(km_gt))))
        kg_rectified_rms_shape = np.sqrt(
                                    np.nanmean(
                                        np.square((kg_pred - kg_gt) / np.maximum(np.abs(kg_gt), np.ones_like(kg_gt))
                                        )
                                    )
                                )
        km_rectified_rms_shape = np.sqrt(
                                    np.nanmean(
                                        np.square((np.abs(km_pred) - np.abs(km_gt)) / np.maximum(np.abs(km_gt), np.ones_like(km_gt))
                                        )
                                    )
                                )
      
        # error metrics
        pcurv_rms.append(pcurv_rms_shape)
        pcurv_rectified_rms.append(rectified_pcurv_rms_shape)
        kg_rms.append(kg_rms_shape)
        km_rms.append(km_rms_shape)
        kg_rectified_rms.append(kg_rectified_rms_shape)
        km_rectified_rms.append(km_rectified_rms_shape)
        
   
    avg_kg_rectified_rms = np.mean(kg_rectified_rms)
    avg_km_rectified_rms = np.mean(km_rectified_rms)
   
    log_string('kg RMS per shape: ' + str(kg_rectified_rms))
    log_string('kg average RMS: ' + str(avg_kg_rectified_rms) + '\n')

    log_string('km RMS per shape: ' + str(km_rectified_rms))
    log_string('km average RMS: ' + str(avg_km_rectified_rms) + '\n')
   
    return avg_kg_rectified_rms, avg_km_rectified_rms    


def test(ckpt_dir, ckpt_iter):
    ### Input/Output
    ckpt_path = os.path.join(args.log_root, ckpt_dir, 'ckpts/ckpt_%s.pt' % ckpt_iter)
    output_dir = os.path.join(args.log_root, ckpt_dir, 'results_%s/ckpt_%s' % (args.data_set, ckpt_iter))
    if args.tag is not None and len(args.tag) != 0:
        output_dir += '_' + args.tag
    if not os.path.exists(ckpt_path):
        print('ERROR path: %s' % ckpt_path)
        return False, False

    file_save_dir = os.path.join(output_dir, 'differential_properties')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(file_save_dir, exist_ok=True)

    pwn_save_dir = os.path.join(file_save_dir, 'PWN')
    os.makedirs(pwn_save_dir, exist_ok=True)

    logger = get_logger('test(%d)(%s-%s)' % (PID, ckpt_dir, ckpt_iter), output_dir)
    logger.info('Command: {}'.format(' '.join(sys.argv)))

    ### Model
    logger.info('Loading model: %s' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GraphCNN(num_points=args.patch_size, use_feat_stn=True).to(device)
    qsurf = QuadraticSurface().to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Num_params: %d' % num_params)
    logger.info('Num_params_trainable: %d' % trainable_num)

    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    shape_ind = 0
    shape_patch_offset = 0
    shape_num = len(test_dset.shape_names)
    shape_patch_count = test_dset.shape_patch_count[shape_ind]

    num_batch = len(test_dataloader)
    normal_prop = torch.zeros([shape_patch_count, 3])
    pcurv_prop = torch.zeros([shape_patch_count, 2])
    kg_prop = torch.zeros([shape_patch_count,])
    km_prop = torch.zeros([shape_patch_count,])

    total_time = 0
    for batchind, batch in enumerate(test_dataloader, 0):
        points_patch = batch['points_patch'].to(device)          # (B, N, 3)
        pca_trans = batch['pca_trans'].to(device)
        patch_radius = batch['patch_radius'].to(device)
        
        start_time = time.time()
        with torch.no_grad():
            weights, offsets = model(points_patch.transpose(2, 1))
            u = qsurf.fit(weights, points_patch + offsets.transpose(2, 1))

            n_est = qsurf.get_origin_normal(u)
            kg_est, km_est = qsurf.get_origin_gaussian_mean_curvatures(u)
            k1, k2 = qsurf.get_principal_curvatures_from_gaussian_mean(kg_est, km_est)
            pcurv_est = torch.cat([k1.unsqueeze(-1), k2.unsqueeze(-1)], dim=-1)

        if pca_trans is not None:
            ### transform predictions with inverse PCA rotation (back to world space)
            n_est = torch.bmm(n_est.unsqueeze(1), pca_trans.transpose(2, 1)).squeeze()
            pass

        if patch_radius is not None:
            pcurv_est = pcurv_est / patch_radius.unsqueeze(-1)
            kg_est = kg_est / patch_radius ** 2
            km_est = km_est / patch_radius

        end_time = time.time()
        elapsed_time = 1000 * (end_time - start_time)  # ms
        total_time += elapsed_time

        if batchind % 5 == 0:
            batchSize = points_patch.size()[0]
            logger.info('[%d/%d] %s: elapsed_time per point/patch: %.3f ms' % (
                        batchind, num_batch-1, test_dset.shape_names[shape_ind], elapsed_time / batchSize))

        ### Save the estimated normals and curvatures to file
        batch_offset = 0
        while batch_offset < n_est.shape[0] and shape_ind + 1 <= shape_num:
            shape_patches_remaining = shape_patch_count - shape_patch_offset
            batch_patches_remaining = n_est.shape[0] - batch_offset

            ### append estimated patch properties batch to properties for the current shape on the CPU
            normal_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining), :] = \
                n_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]
            pcurv_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining), :] = \
                pcurv_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining), :]
            kg_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)] = \
                kg_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining)]
            km_prop[shape_patch_offset:shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)] = \
                km_est[batch_offset:batch_offset + min(shape_patches_remaining, batch_patches_remaining)]

            batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
            shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

            if shape_patches_remaining <= batch_patches_remaining:
                normals_to_write = normal_prop.cpu().numpy()
                pcurv_to_write = pcurv_prop.cpu().numpy()
                kg_to_write = kg_prop.cpu().numpy()
                km_to_write = km_prop.cpu().numpy()

                ### for faster reading speed in the evaluation
                save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_normal.npy')
                np.save(save_path, normals_to_write)
                if args.save_pn:
                    save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '.normals')
                    np.savetxt(save_path, normals_to_write)
                logger.info('saved normal: {} \n'.format(save_path))

                save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_curv.npy')
                np.save(save_path, pcurv_to_write)
                if args.save_pcurv:
                    save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '.curv')
                    np.savetxt(save_path, pcurv_to_write)
                logger.info('saved principal curvatures: {} \n'.format(save_path))

                save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_kg.npy')
                np.save(save_path, kg_to_write)
                if args.save_kg:
                    save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '.kg')
                    np.savetxt(save_path, kg_to_write)
                logger.info('saved gaussian curvature: {} \n'.format(save_path))

                save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '_km.npy')
                np.save(save_path, km_to_write)
                if args.save_km:
                    save_path = os.path.join(file_save_dir, test_dset.shape_names[shape_ind] + '.km')
                    np.savetxt(save_path, km_to_write)
                logger.info('saved mean curvature: {} \n'.format(save_path))

                #save pwn
                save_pwn_path = os.path.join(pwn_save_dir, test_dset.shape_names[shape_ind] + '.xyz')
                xyz_input_path = os.path.join(os.path.join(args.dataset_root, args.data_set), test_dset.shape_names[shape_ind] + '.xyz')
                xyz_input = np.loadtxt(xyz_input_path)  
                if args.sparse_patches:
                    pids_path = os.path.join(os.path.join(args.dataset_root, args.data_set), test_dset.shape_names[shape_ind] + '.pidx')
                    points_idx = np.loadtxt(pids_path,dtype=np.int32)      # (n,)
                    xyz_input = xyz_input[points_idx, :]
                pwn_output = np.concatenate((xyz_input, normals_to_write), axis=1) 
                np.savetxt(save_pwn_path, pwn_output, fmt='%.6f')                              

                sys.stdout.flush()
                shape_patch_offset = 0
                shape_ind += 1
                if shape_ind < shape_num:
                    shape_patch_count = test_dset.shape_patch_count[shape_ind]
                    normal_prop = torch.zeros([shape_patch_count, 3])
                    curv_prop = torch.zeros([shape_patch_count, 2])

    logger.info('Total Time: %.2f s, Shape Num: %d' % (total_time/1000, shape_num))
    return output_dir, file_save_dir


def eval(normal_gt_path, normal_pred_path, pcurv_gt_path, pcurv_pred_path, kg_pred_path, km_pred_path, output_dir):
    print('\n  Evaluation ...')
    eval_summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(eval_summary_dir, exist_ok=True)
    
    all_avg_n_rms, all_avg_kg_rms, all_avg_km_rms = [], [], []
    for cur_list in args.eval_list:
        print("\n***************** " + cur_list + " *****************")
        print("Result path: " + normal_pred_path)

        ### get all shape names in the list
        shape_names = []
        normal_gt_filenames = os.path.join(normal_gt_path, 'list', cur_list)
        with open(normal_gt_filenames) as f:
            shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))

        ### load all shapes
        normal_gts = []
        normal_preds = []
        pcurv_gts = []
        pcurv_preds = []
        kg_gts = []
        kg_preds = []
        km_gts = []
        km_preds = []
        for shape in shape_names:
            print(shape)
            normal_gt = load_data(filedir=normal_gt_path, filename=shape + '.normals', dtype=np.float32)  # (N, 3)
            normal_pred = np.load(os.path.join(normal_pred_path, shape + '_normal.npy'))                  # (n, 3)
            ### eval with sparse point sets
            points_idx = load_data(filedir=normal_gt_path, filename=shape + '.pidx', dtype=np.int32)      # (n,)
            normal_gt = normal_gt[points_idx, :]
            if normal_pred.shape[0] > normal_gt.shape[0]:
                normal_pred = normal_pred[points_idx, :]

            normal_gts.append(normal_gt)
            normal_preds.append(normal_pred)

            pcurv_gt = load_data(filedir=pcurv_gt_path, filename=shape + '.curv', dtype=np.float32)  # (N, 2)
            kg_gt = pcurv_gt[:, 0] * pcurv_gt[:, 1]
            km_gt = 0.5 * (pcurv_gt[:, 0] + pcurv_gt[:, 1])
            pcurv_pred = np.load(os.path.join(pcurv_pred_path, shape + '_curv.npy'))                  # (n, 2)
            pcurv_gt = pcurv_gt[points_idx, :]
            if pcurv_pred.shape[0] > pcurv_gt.shape[0]:
                pcurv_pred = pcurv_pred[points_idx, :]

            pcurv_gts.append(pcurv_gt)
            pcurv_preds.append(pcurv_pred)

            kg_pred = np.load(os.path.join(kg_pred_path, shape + '_kg.npy'))  
            kg_gt = kg_gt[points_idx]
            if kg_pred.shape[0] > kg_gt.shape[0]:
                kg_pred = kg_pred[points_idx]

            kg_gts.append(kg_gt)
            kg_preds.append(kg_pred)

            km_pred = np.load(os.path.join(kg_pred_path, shape + '_km.npy'))  
            km_gt = km_gt[points_idx]
            if km_pred.shape[0] > km_gt.shape[0]:
                km_pred = km_pred[points_idx]

            km_gts.append(km_gt)
            km_preds.append(km_pred)

        ### compute RMSE per-list
        avg_n_rms = normal_RMSE(normal_gts=np.array(normal_gts),
                            normal_preds=np.array(normal_preds),
                            eval_file=os.path.join(eval_summary_dir, cur_list[:-4] + '_evaluation_results.txt'))
        all_avg_n_rms.append(avg_n_rms)
        print('normal RMSE: %f' % avg_n_rms)

        avg_kg_rms, avg_km_rms  = curvature_RMSE(pcurv_gts=np.array(pcurv_gts), 
                            pcurv_preds=np.array(pcurv_preds),
                            kg_gts=np.array(kg_gts),
                            kg_preds=np.array(kg_preds),
                            km_gts=np.array(km_gts),
                            km_preds=np.array(km_preds),
                            eval_file=os.path.join(eval_summary_dir, cur_list[:-4] + '_evaluation_results.txt'))
        all_avg_kg_rms.append(avg_kg_rms)
        all_avg_km_rms.append(avg_km_rms)
        print('kg RMSE: %f, km RMSE %f' % (avg_kg_rms, avg_km_rms))

    s = '\n {} \n All RMS not oriented (shape average): {} | Mean: {}\n'.format(
                normal_pred_path, str(all_avg_n_rms), np.mean(all_avg_n_rms))
    print(s)

    ## delete the output point normals
    if not args.save_pn:
        shutil.rmtree(normal_pred_path)

    return all_avg_n_rms, all_avg_kg_rms, all_avg_km_rms



if __name__ == '__main__':
    ckpt_dirs = args.ckpt_dirs.split(',')
    ckpt_iter = args.ckpt_iter

    for ckpt_dir in ckpt_dirs:
        eval_dict = ''
        sum_file = 'eval_' + args.data_set + ('_'+args.tag if len(args.tag) != 0 else '')
        log_file_sum = open(os.path.join(args.log_root, ckpt_dir, sum_file+'.txt'), 'a')
        log_file_sum.write('\n====== %s ======\n' % args.eval_list)
     
        output_dir, file_save_dir = test(ckpt_dir=ckpt_dir, ckpt_iter=ckpt_iter)

        all_avg_n_rms, all_avg_kg_rms, all_avg_km_rms = eval(normal_gt_path=os.path.join(args.dataset_root, args.data_set),
                           normal_pred_path=file_save_dir,
                           pcurv_gt_path=os.path.join(args.dataset_root, args.data_set),
                           pcurv_pred_path=file_save_dir,
                           kg_pred_path=file_save_dir,
                           km_pred_path=file_save_dir,
                           output_dir=output_dir)

        s = '%s: %s | Noraml Metric Mean: %f\n' % (ckpt_iter, str(all_avg_n_rms), np.mean(all_avg_n_rms))
        log_file_sum.write(s)
        log_file_sum.flush()
        eval_dict += s


        all_avg_kg_rms = np.array(all_avg_kg_rms)
        s = '%s: %s | kg Metric Mean: %f \n' % (ckpt_iter, str(all_avg_kg_rms), np.mean(all_avg_kg_rms))
        log_file_sum.write(s)
        log_file_sum.flush()
        eval_dict += s

        all_avg_km_rms = np.array(all_avg_km_rms)
        s = '%s: %s | km Metric Mean: %f \n' % (ckpt_iter, str(all_avg_km_rms), np.mean(all_avg_km_rms))
        log_file_sum.write(s)
        log_file_sum.flush()
        eval_dict += s

        log_file_sum.close()
        s = '\n All RMS not oriented (shape average): \n{}\n'.format(eval_dict)
        print(s)
      


