import os, time, torch, itertools, warnings
from config import parse_args
from data_loader import sample_dataset, get_loader
from model import ResnetBlock, Decoder, Discriminator
import torch.nn as nn
from torchvision import transforms
from glob import glob
from se50_net import se50_net
warnings.filterwarnings("ignore")
import numpy as np
#for demo
import cv2
from util import *

epsilon = 1e-9

class FNM(object):
    def __init__(self, args):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.profile_list_path = args.profile_list
        self.front_list_path = args.front_list
        self.profile_path = args.profile_path
        self.front_path = args.front_path
        self.test_path = args.test_path
        self.test_list = args.test_list

        self.crop_size = args.ori_height
        self.image_size = args.height
        self.res_n = args.res_n
        self.is_finetune = args.is_finetune
        self.result_name = args.result_name
        self.summary_dir = args.summary_dir
        self.iteration = args.iteration
        self.weight_decay = args.weight_decay
        self.decay_flag = args.decay_flag
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.img_size = args.width
        self.model_name = args.model_name

        # For hyper parameters
        self.lambda_l1 = args.lambda_l1
        self.lambda_fea = args.lambda_fea
        self.lambda_reg = args.lambda_reg
        self.lambda_gan = args.lambda_gan
        self.lambda_gp = args.lambda_gp

        self.channel = args.channel
        self.device = torch.device("cuda:{}".format(args.device_id))
        self.make_dirs()
        self.build_model()

        """Define Loss"""
        self.L1_loss = nn.L1Loss().to(self.device)
        self.L2_loss = nn.MSELoss().to(self.device)

    def make_dirs(self):
        check_folder(self.summary_dir)
        check_folder(os.path.join("results", self.result_name, "model"))
        check_folder(os.path.join("results", self.result_name, "img"))
    
    def build_model(self):
        self.expert_net = se50_net("./other_models/arcface_se50/model_ir_se50.pth").to(self.device)
        for param in self.expert_net.parameters():
            param.requires_grad = False
        #self.dataset = sample_dataset(self.profile_list_path, self.front_list_path, self.profile_path, self.front_path, self.crop_size, self.image_size)
        self.front_loader = get_loader(self.front_list_path, self.front_path, self.crop_size, self.image_size, self.batch_size, mode="train", num_workers=8)

        self.profile_loader = get_loader(self.profile_list_path, self.profile_path, self.crop_size, self.image_size, self.batch_size, mode="train", num_workers=8)

        self.test_loader = get_loader(self.test_list, self.test_path, self.crop_size, self.image_size, self.batch_size, mode="test", num_workers=8)

        #self.front_loader = iter(self.front_loader)
        #self.profile_loader = iter(self.profile_loader)
        #resnet_blocks
        resnet_block_list = []
        for i in range(self.res_n):
            resnet_block_list.append(ResnetBlock(512, use_bias=False))
        
        self.body = nn.Sequential(*resnet_block_list).to(self.device)
        #[b, 512, 7, 7]
        self.decoder = Decoder().to(self.device)
        self.dis = Discriminator(self.channel).to(self.device)

        self.G_optim = torch.optim.Adam(itertools.chain(self.body.parameters(), self.decoder.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        self.D_optim = torch.optim.Adam(itertools.chain(self.dis.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        self.downsample112x112 = nn.Upsample(size=(112, 112), mode='bilinear')

    def update_lr(self, start_iter):
        if self.decay_flag and start_iter > (self.iteration // 2):
            self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
            self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

    def train(self):
        self.body.train(), self.decoder.train(), self.dis.train()
        start_iter = 1
        if self.is_finetune:
            model_list = glob(os.path.join("results", self.result_name, "model", "*.pt"))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join("results", self.result_name, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                self.update_lr(start_iter)
        print("training start...")
        start_time = time.time()
        for step in range(start_iter, self.iteration+1):
            self.update_lr(start_iter)
            try:
                front_224, front_112 = front_iter.next()
                if front_224.shape[0] != self.batch_size:
                    raise Exception
            except:
                front_iter = iter(self.front_loader)
                front_224, front_112 = front_iter.next()
            try:
                profile_224, profile_112 = profile_iter.next()
                if profile_224.shape[0] != self.batch_size:
                    raise Exception
            except:
                profile_iter = iter(self.profile_loader)
                profile_224, profile_112 = profile_iter.next()

            profile_224, front_224, profile_112, front_112 = profile_224.to(self.device), front_224.to(self.device), profile_112.to(self.device), front_112.to(self.device)

            # Update D
            self.D_optim.zero_grad()

            feature_p = self.expert_net.get_feature(profile_112)
            feature_f = self.expert_net.get_feature(front_112)
            gen_p = self.decoder(self.body(feature_p))
            gen_f = self.decoder(self.body(feature_f))
            feature_gen_p = self.expert_net.get_feature(self.downsample112x112(gen_p))
            feature_gen_f = self.expert_net.get_feature(self.downsample112x112(gen_f))
            d_f = self.dis(front_224)
            d_gen_p = self.dis(gen_p)
            d_gen_f = self.dis(gen_f)
            
            D_adv_loss = torch.mean(tensor_tuple_sum(d_gen_f)*0.5 + tensor_tuple_sum(d_gen_p)*0.5 - tensor_tuple_sum(d_f))/5

            alpha = torch.rand(gen_p.size(0), 1, 1, 1).to(self.device)
            inter = (alpha * front_224.data + (1 - alpha) * gen_p.data).requires_grad_(True)
            out_inter = self.dis(inter)
            gradient_penalty_loss = (gradient_penalty(out_inter[0], inter, self.device) + gradient_penalty(out_inter[1], inter, self.device) + gradient_penalty(out_inter[2], inter, self.device) + gradient_penalty(out_inter[3], inter, self.device))/4
            #print("gradient_penalty_loss:{}".format(gradient_penalty_loss))
            d_loss = self.lambda_gan*D_adv_loss + self.lambda_gp*gradient_penalty_loss
            d_loss.backward(retain_graph=True)
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()
            try:
                front_224, front_112 = front_iter.next()
                if front_224.shape[0] != self.batch_size:
                    raise Exception
            except:
                front_iter = iter(self.front_loader)
                front_224, front_112 = front_iter.next()
            try:
                profile_224, profile_112 = profile_iter.next()
                if profile_224.shape[0] != self.batch_size:
                    raise Exception
            except:
                profile_iter = iter(self.profile_loader)
                profile_224, profile_112 = profile_iter.next()

            profile_224, front_224, profile_112, front_112 = profile_224.to(self.device), front_224.to(self.device), profile_112.to(self.device), front_112.to(self.device)
            
            feature_p = self.expert_net.get_feature(profile_112)
            feature_f = self.expert_net.get_feature(front_112)
            gen_p = self.decoder(self.body(feature_p))
            gen_f = self.decoder(self.body(feature_f))
            feature_gen_p = self.expert_net.get_feature(self.downsample112x112(gen_p))
            feature_gen_f = self.expert_net.get_feature(self.downsample112x112(gen_f))
            d_f = self.dis(front_224)
            d_gen_p = self.dis(gen_p)
            d_gen_f = self.dis(gen_f)

            pixel_loss = torch.mean(self.L1_loss(front_224, gen_f))

            feature_p_norm = l2_norm(feature_p)
            feature_f_norm = l2_norm(feature_f)
            feature_gen_p_norm = l2_norm(feature_gen_p)
            feature_gen_f_norm = l2_norm(feature_gen_f)

            perceptual_loss = torch.mean(0.5*(1-torch.sum(torch.mul(feature_p_norm, feature_gen_p_norm), dim=(1, 2, 3))) + 0.5*(1-torch.sum(torch.mul(feature_f_norm, feature_gen_f_norm), dim=(1, 2, 3))))
            
            G_adv_loss = -torch.mean(tensor_tuple_sum(d_gen_f)*0.5 + tensor_tuple_sum(d_gen_p)*0.5)/5
            g_loss = self.lambda_gan*G_adv_loss + self.lambda_l1*pixel_loss + self.lambda_fea*perceptual_loss
            g_loss.backward()
            self.G_optim.step()
            
            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, d_loss, g_loss))
            print("D_adv_loss : %.8f" % (self.lambda_gan*D_adv_loss))
            print("G_adv_loss : %.8f" % (self.lambda_gan*G_adv_loss))
            print("pixel_loss : %.8f" % (self.lambda_l1*pixel_loss))
            print("perceptual_loss : %.8f" % (self.lambda_fea*perceptual_loss))
            print("gp_loss : %.8f" % (self.lambda_gp*gradient_penalty_loss))

            with torch.no_grad():
                if step % self.print_freq == 0:
                    train_sample_num = 5
                    test_sample_num = 5
                    A2B = np.zeros((self.img_size * 4, 0, 3))
                    self.body.eval(), self.decoder.eval(), self.dis.eval()
                    for _ in range(train_sample_num):
                        try:
                            front_224, front_112 = front_iter.next()
                            if front_224.shape[0] != self.batch_size:
                                raise Exception
                        except:
                            front_iter = iter(self.front_loader)
                            front_224, front_112 = front_iter.next()
                        try:
                            profile_224, profile_112 = profile_iter.next()
                            if profile_224.shape[0] != self.batch_size:
                                raise Exception
                        except:
                            profile_iter = iter(self.profile_loader)
                            profile_224, profile_112 = profile_iter.next()

                        profile_224, front_224, profile_112, front_112 = profile_224.to(self.device), front_224.to(self.device), profile_112.to(self.device), front_112.to(self.device)

                        feature_p = self.expert_net.get_feature(profile_112)
                        feature_f = self.expert_net.get_feature(front_112)
                        gen_p = self.decoder(self.body(feature_p))
                        gen_f = self.decoder(self.body(feature_f))

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(profile_224[0]))), RGB2BGR(tensor2numpy(denorm(gen_p[0]))), RGB2BGR(tensor2numpy(denorm(front_224[0]))), RGB2BGR(tensor2numpy(denorm(gen_f[0])))), 0)), 1)

                    for _ in range(train_sample_num):
                        show_list = []
                        for i in range(2):
                            try:
                                test_profile_224, test_profile_112 = test_iter.next()
                                if test_profile_224.shape[0] != self.batch_size:
                                    raise Exception
                            except:
                                test_iter = iter(self.test_loader)
                                test_profile_224, test_profile_112 = test_iter.next()
                            test_profile_224, test_profile_112 = test_profile_224.to(self.device), test_profile_112.to(self.device)
                            test_feature_p = self.expert_net.get_feature(test_profile_112)
                            test_gen_p = self.decoder(self.body(test_feature_p))
                            show_list.append(test_profile_224[0])
                            show_list.append(test_gen_p[0])

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(show_list[0]))), RGB2BGR(tensor2numpy(denorm(show_list[1]))), RGB2BGR(tensor2numpy(denorm(show_list[2]))), RGB2BGR(tensor2numpy(denorm(show_list[3])))), 0)), 1)
                        
                    cv2.imwrite(os.path.join("results", self.result_name, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    self.body.train(), self.decoder.train(), self.dis.train()

                if step % self.save_freq == 0:
                    self.save(os.path.join("results", self.result_name, "model"), step)

                if step % 1000 == 0:
                    params = {}
                    params['body'] = self.body.state_dict()
                    params['decoder'] = self.decoder.state_dict()
                    params['dis'] = self.dis.state_dict()
                    torch.save(params, os.path.join("results", self.result_name, self.model_name+"_params_latest.pt"))
    
    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.model_name + '_params_%07d.pt' % step))
        self.body.load_state_dict(params['body'])
        self.decoder.load_state_dict(params['decoder'])
        self.dis.load_state_dict(params['dis'])

    def save(self, dir, step):
        params = {}
        params['body'] = self.body.state_dict()
        params['decoder'] = self.decoder.state_dict()
        params['dis'] = self.dis.state_dict()
        torch.save(params, os.path.join(dir, self.model_name + '_params_%07d.pt' % step))



        
    def demo(self):
        try:
            front_224, front_112 = front_iter.next()
            if front_224.shape[0] != self.batch_size:
                raise Exception
        except:
            front_iter = iter(self.front_loader)
            front_224, front_112 = front_iter.next()
        try:
            profile_224, profile_112 = profile_iter.next()
            if profile_224.shape[0] != self.batch_size:
                raise Exception
        except:
            profile_iter = iter(self.profile_loader)
            profile_224, profile_112 = profile_iter.next()

        profile_224, front_224, profile_112, front_112 = profile_224.to(self.device), front_224.to(self.device), profile_112.to(self.device), front_112.to(self.device)
        D_face, D_eye, D_nose, D_mouth = self.dis(profile_224)
        
        '''
        print("D_face.shape:", D_face.shape)
        print("D_eye.shape:", D_eye.shape)
        print("D_nose.shape:", D_nose.shape)
        print("D_mouth.shape:", D_mouth.shape)
        '''
        cv2.imwrite("profile.jpg", cv2.cvtColor(tensor2im(profile_112), cv2.COLOR_BGR2RGB))
        cv2.imwrite("front.jpg", cv2.cvtColor(tensor2im(front_112), cv2.COLOR_BGR2RGB))
        feature = self.expert_net.get_feature(profile_224)
        print(feature.shape)
        '''
        feature = self.body(feature)
        out = self.decoder(feature)
        print(out.shape)
        '''


if __name__ == '__main__':
    args = parse_args()
    model = FNM(args)
    model.train()
    #model.demo()
        

