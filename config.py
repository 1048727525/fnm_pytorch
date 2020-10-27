import argparse
from util import *
def parse_args():
    parser = argparse.ArgumentParser()

    ############################
    #    hyper parameters      #
    ############################

    # For hyper parameters
    parser.add_argument('--lambda_l1', type=float, default=0.001, help='weight of the loss for L1 texture loss')
    parser.add_argument('--lambda_fea', type=float, default=100, help='weight of the loss for face model feature loss')
    parser.add_argument('--lambda_reg', type=float, default=1e-5, help='weight of the loss for L2 regularitaion loss')
    parser.add_argument('--lambda_gan', type=float, default=1, help='weight of the loss for gan loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight of the loss for gradient penalty on parameter of D')

    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of model save freq')

    # For training
    parser.add_argument('--dataset_size', type=int, default=297369, help='number of non-normal face set')
    '''
    parser.add_argument('--profile_path', type=str, default='../../datasets/casia_aligned_250_250_jpg', help='dataset path')
    parser.add_argument('--profile_list', type=str, default='../fnm/mpie/casia_profile.txt', help='train profile list')
    parser.add_argument('--front_path', type=str, default='../../datasets/session01_align', help='front data path')
    parser.add_argument('--front_list', type=str, default='../fnm/mpie/session01_front.txt', help='train front list')
    parser.add_argument('--test_path', type=str, default='../../datasets/session01_align', help='front data path')
    parser.add_argument('--test_list', type=str, default='../fnm/mpie/session01_profile.txt', help='test front list')
    '''

    parser.add_argument('--profile_path', type=str, default='../../datasets/session01_align', help='dataset path')
    parser.add_argument('--profile_list', type=str, default='../fnm/mpie/setting1_profile.txt', help='train profile list')
    parser.add_argument('--front_path', type=str, default='../../datasets/session01_align', help='front data path')
    parser.add_argument('--front_list', type=str, default='../fnm/mpie/setting1_front.txt', help='train front list')
    parser.add_argument('--test_path', type=str, default='../../datasets/session01_align', help='front data path')
    parser.add_argument('--test_list', type=str, default='../fnm/mpie/setting1_test.txt', help='test front list')

    parser.add_argument('--is_train', type=bool, default=True, help='train or test')
    parser.add_argument('--is_finetune', type=bool, default=False, help='finetune')

    parser.add_argument('--result_name', type=str, default='v0', help='result directory')
    parser.add_argument('--summary_dir', type=str, default='log', help='logs directory')
    parser.add_argument('--checkpoint_ft', type=str, default='checkpoint/fnm/ck-09', help='finetune or test checkpoint path')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    parser.add_argument('--critic', type=int, default=1, help='number of D training times')
    parser.add_argument('--iteration', type=int, default=300000, help='The number of training iterations')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--lr', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--stddev', type=float, default=0.02, help='stddev for W initializer')
    parser.add_argument('--use_bias', type=bool, default=False, help='whether to use bias')
    parser.add_argument('--norm', type=str, default='bn', help='normalize function for G')
    parser.add_argument('--results', type=str, default='results/fnm', help='path for saving results')
    parser.add_argument('--res_n', type=int, default=4, help='number of resnet_block')
    parser.add_argument('--model_name', type=str, default='fnm', help='model name')

    ############################
    #   environment setting    #
    ############################
    parser.add_argument('--device_id', type=str, default='0', help='device id')
    parser.add_argument('--ori_height', type=int, default=224, help='original height of profile images')
    parser.add_argument('--ori_width', type=int, default=224, help='original width of profile images')
    parser.add_argument('--height', type=int, default=224, help='height of images')
    parser.add_argument('--width', type=int, default=224, help='width of images')
    parser.add_argument('--channel', type=int, default=3, help='channel of images')
    parser.add_argument('--num_threads', type=int, default=8, help='number of threads of enqueueing examples')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()