import argparse
import ml_collections
parser = argparse.ArgumentParser(description='Hyper-parameters management')

# # Hardware options
# parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
# parser.add_argument('--cpu', action='store_true',help='use cpu only')
# parser.add_argument('--gpu_id', type=list,default=[0,1], help='use cpu only')
# parser.add_argument('--seed', type=int, default=123, help='random seed')
#
# # Preprocess parameters
# parser.add_argument('--n_labels', type=int, default=2,help='number of classes') # 分割肝脏则置为2（二类分割），分割肝脏和肿瘤则置为3（三类分割）
# parser.add_argument('--upper', type=int, default=200, help='')
# parser.add_argument('--lower', type=int, default=-200, help='')
# parser.add_argument('--norm_factor', type=float, default=200.0, help='')
# parser.add_argument('--expand_slice', type=int, default=20, help='')
# parser.add_argument('--min_slices', type=int, default=48, help='')
# parser.add_argument('--xy_down_scale', type=float, default=0.5, help='')
# parser.add_argument('--slice_down_scale', type=float, default=1.0, help='')
# parser.add_argument('--valid_rate', type=float, default=0.2, help='')
#
# # data in/out and dataset
# parser.add_argument('--dataset_path',default = '/ssd/lzq/dataset/fixed_lits',help='fixed trainset root path')
# parser.add_argument('--test_data_path',default = '/ssd/lzq/dataset/LiTS/test',help='Testset path')
# parser.add_argument('--save',default='temp_4',help='save path of trained model')
# parser.add_argument('--batch_size', type=list, default=1,help='batch size of trainset')
#
# # train
#
# parser.add_argument('--epochs', type=int, default=500, metavar='N',help='number of epochs to train (default: 200)')
# parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.0001)')
# parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 30)')
# parser.add_argument('--crop_size', type=int, default=48)
# parser.add_argument('--val_crop_max_size', type=int, default=96)
#
# # test
# parser.add_argument('--test_cut_size', type=int, default=48, help='size of sliding window')
# parser.add_argument('--test_cut_stride', type=int, default=24, help='stride of sliding window')
# parser.add_argument('--postprocess', type=bool, default=False, help='post process')
parser.add_argument('--cpu',default=False ,action='store_true',help='use cpu only')
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--epochs', type=int,
                    default=250, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-L_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

args = parser.parse_args()
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config

