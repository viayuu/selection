import os
import random

import numpy as np
import torch
from logging import getLogger

from Env.SDVRPEnv import SDVRPEnv
from Env.TSPEnv import TSPEnv
from Env.CVRPEnv import CVRPEnv
from Env.PCTSPEnv import PCTSPEnv
from Env.OPEnv import OPEnv
from Env.SPCTSPEnv import SPCTSPEnv
from Env.ATSPEnv import ATSPEnv
from Env.PDPEnv import PDPEnv
from Env.VRPMIXEnv import VRPMIXEnv
from Env.ACVRPEnv import ACVRPEnv

from Model_RELD import Model as Model

from multi_hot_set import *

from torch.optim import AdamW

from utils.utils import *
from ProblemDef import get_saved_data


class Trainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 train_problem_list,
                 test_problem_list=None):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.train_problem_list = train_problem_list
        if test_problem_list is None:
            self.test_problem_list = train_problem_list
        else:
            self.test_problem_list = test_problem_list



        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # 使用集合求两个列表的并集
        self.all_problems = list(set(train_problem_list).union(set(test_problem_list)))

        self.logger.info("Training problem list(num: {}): {}".format(len(self.train_problem_list), self.train_problem_list))
        self.logger.info("Testing problem list(num: {}): {}".format(len(self.test_problem_list), self.test_problem_list))

        self.problem_attributes_set = get_problem_attributes_set()

        self.env_classes = {
                            "tsp": TSPEnv,
                            "cvrp": CVRPEnv,
                            "op": OPEnv,
                            "pctsp":PCTSPEnv,
                            "sdvrp":SDVRPEnv,
                            "spctsp":SPCTSPEnv,
                            "atsp":ATSPEnv,
                            "pdp": PDPEnv,
                            "acvrp":ACVRPEnv,
                            }
        vrpmix_variants = [
            "cvrptw", "ovrp", "vrpl", "vrpb", "ovrptw",
            "ovrpb", "vrpbl", "vrpltw", "ovrpbtw", "vrpbltw",
            "ovrpl", "vrpbtw", "ovrpbl", "ovrpltw", "ovrpbltw"
        ]
        for k in vrpmix_variants:
            self.env_classes[k] = VRPMIXEnv

        self.problem_gaps = {name: {100: 100.0, 1000: 100.0} for name in test_problem_list}

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.device = device

        # Main Components
        self.model = Model(**self.model_params)
        self.optimizer = AdamW(self.model.parameters(), **self.optimizer_params['optimizer'])

        # if self.optimizer_params['optimizer_type'] == 'AdamW':
        #     self.optimizer = AdamW(self.model.parameters(), **self.optimizer_params['optimizer'])
        # elif self.optimizer_params['optimizer_type'] == 'Adam':
        #     self.optimizer = Adam(self.model.parameters(), **self.optimizer_params['optimizer'])
        # else:
        #     raise NotImplementedError(f"Optimizer type {self.optimizer_params['optimizer_type']} is not implemented!")

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Saved Model Loaded !!')
            self.logger.info("Model loaded from: {}".format(checkpoint_fullname))

        # utility
        self.time_estimator = TimeEstimator()

        self.problem_validation_set = get_problem_validation_set()

        self.saved_data = {}
        self.optimal_scores = {}
        validation_scale_list = [100]
        for scale in validation_scale_list:
            # 2. 遍历问题配置列表，加载数据
            self.saved_data[scale] = {}
            self.optimal_scores[scale] = {}
            for problem_name in self.test_problem_list:

                detailed_validation = self.problem_validation_set[problem_name]

                num_instances = detailed_validation.get(f"episodes_{scale}", 10000)  # 默认10000个实例

                filename = os.path.join("data", problem_name, detailed_validation.get(f"filename_{scale}", ""))
                extra_args = detailed_validation.get("extra_args", {})
                saved_data, optimal_score = get_saved_data(
                    filename,
                    problem_name,
                    num_instances,
                    self.device,
                    solution_name= extra_args.get(f"solution_{scale}", None)
                )

                self.saved_data[scale][problem_name] = saved_data
                self.optimal_scores[scale][problem_name] = optimal_score

                self.logger.info('Successfully load {0:5d} {1:10s} instances, optimal_score: {2:7.4f}'.format(
                    num_instances, problem_name.upper()+str(scale), optimal_score))

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        self.lr_decay_epoch = self.optimizer_params['lr_decay_epoch']

        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('========================================================================')

            #########################################################################
            # Train
            #########################################################################
            if epoch in self.lr_decay_epoch:
                self.optimizer.param_groups[0]['lr'] = self.optimizer_params['optimizer']['lr'] * 0.1 # 1e-5
            self.logger.info('Epoch {:4d}: Current learning rate: {}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            train_loss = self._train_one_epoch(epoch)

            self.result_log.append('train_loss', epoch, train_loss)

            #########################################################################
            # Validation
            #########################################################################
            gap_log_parts = []
            for problem_type in self.test_problem_list:
                saved_100 = self.saved_data[100][problem_type]
                optimal_score_100 = self.optimal_scores[100][problem_type]
                # saved_1000 = self.saved_data[1000][problem_type]
                # optimal_score_1000 = self.optimal_scores[1000][problem_type]
                # self._validation_one_epoch(problem_type, saved_100, optimal_score_100, epoch,
                #                            self.problem_validation_set[problem_type]['episodes_100'], 100,batch_size=None)
                self._validation_one_epoch(problem_type, saved_100, optimal_score_100, epoch,
                                           1000, 100,
                                           batch_size=None)
                # self._validation_one_epoch(problem_type, saved_1000, optimal_score_1000, epoch, self.problem_validation_set[problem_type]['episodes_1000'], 1000)
                if self.trainer_params['vst_base_batch_size'] == None:
                    gap = self.problem_gaps[problem_type][100]
                    gap_log_parts.append(f"{problem_type.upper()}{100}: {gap:.4f}%")
                else:
                    for size in [100, 1000]:
                        gap = self.problem_gaps[problem_type][size]
                        gap_log_parts.append(f"{problem_type.upper()}{size}: {gap:.4f}%")
            gap_log_str = "Current detailed gaps: " + ", ".join(gap_log_parts)
            self.logger.info(gap_log_str)
            #########################################################################
            # Logs & Checkpoint
            #########################################################################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:4d}/{:4d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()

                }
                torch.save(checkpoint_dict, '{0}/checkpoint-{1}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        loss = AverageMeter()

        batches_per_epoch =  self.trainer_params['batches_per_epoch']
        loop_cnt = 0 # used for counting the number of batches

        while loop_cnt < batches_per_epoch:
            problem_name = random.choice(self.train_problem_list)

            if self.trainer_params['vst_base_batch_size'] == None:
                true_problem_size = self.env_params['min_problem_size']
                true_batch_size = self.trainer_params['stage1_batch_size']
            else:
                if epoch <= self.trainer_params['stage1_epochs']:
                    true_problem_size = self.env_params['min_problem_size']
                    true_batch_size = self.trainer_params['stage1_batch_size']
                else:
                    # varying scale of training samples
                    true_problem_size = np.random.randint(self.env_params['min_problem_size'],
                                                          self.env_params['max_problem_size'] + 1)
                    true_batch_size = int(self.trainer_params['vst_base_batch_size'] * ((100 / true_problem_size) ** 2))

            if problem_name in ["cvrp",'sdvrp','cvrptw','ovrp','vrpl','vrpb','ovrptw','acvrp',
                                'ovrpb','vrpbl','vrpltw','ovrpbtw','vrpbltw','ovrpl','vrpbtw','ovrpbl','ovrpltw','ovrpbltw']:
                # capacity = np.random.randint(self.env_params['min_capacity'],
                #                              self.env_params['max_capacity'] + 1)
                capacity = 50.0
            else:
                capacity = None

            kwargs = {
                "capacity": capacity,
            }
            avg_score, avg_loss = self._train_one_batch(problem_name,true_problem_size,true_batch_size,epoch,kwargs)

            loss.update(avg_loss, true_batch_size)

            loop_cnt += 1
            if loop_cnt <= 5 or loop_cnt % 200 == 0:
                self.logger.info('Epoch {:4d}: Trained batches {:4d}/{:4d}({:5.1f}%), Problem name: {:8s}, Cur_score: {:7.4f},  Cur_loss: {:7.4f}, Loss: {:7.4f}'
                                 .format(epoch, loop_cnt, batches_per_epoch, 100. * loop_cnt / batches_per_epoch, problem_name,
                                         avg_score, avg_loss, loss.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:4d}: Train ({:3.0f}%)  Loss: {:.4f}'
                         .format(epoch, 100. * loop_cnt / batches_per_epoch, loss.avg))

        return loss.avg

    def _train_one_batch(self, problem_name,problem_size,batch_size,epoch,kwargs):

        # Prep
        ###############################################
        self.model.train()
        self.model.set_decoder_type("sampling")
        self.env = self.env_classes[problem_name]()

        if problem_name not in self.all_problems:
            raise NotImplementedError(f"problem_name: {problem_name} is not implemented!")

        problem_representation = torch.tensor(self.problem_attributes_set[problem_name], dtype=torch.float32,device= self.device)

        pomo_size = problem_size
        self.env.load_problems(batch_size,
                               problem_size=problem_size,
                               pomo_size=pomo_size,
                               device=self.device,
                               problem_name=problem_name,
                               **kwargs)

        # vrpb有负demand节点,不能从这些节点出发，在Env.load_problems里会修改pomo_size
        pomo_size = self.env.pomo_size

        reset_state, _, _ = self.env.reset()
        self.model.decoder.assign(problem_representation)
        self.model.pre_forward(reset_state, problem_name, problem_representation)
        prob_list = torch.zeros(size=(batch_size, pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            cur_dist = self.env.get_local_feature()
            selected, prob = self.model(state,cur_dist)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)

        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        if problem_name in ["op"]:
            score_mean = max_pomo_reward.float().mean()
        else:
            score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

    def _validation_one_epoch(self, problem_name, dataset, optimal_score, epoch,episodes,problem_size,batch_size=None):

        self.model.eval()
        self.model.set_decoder_type("greedy")
        self.env = self.env_classes[problem_name]()

        problem_representation = torch.tensor(self.problem_attributes_set[problem_name], dtype=torch.float32,device= self.device)
        tested_episodes = 0
        if batch_size is None:
            batch_size = episodes

        results = torch.zeros(size=(episodes,), dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            while tested_episodes < episodes:
                remaining_episodes = episodes - tested_episodes
                batch_size = min(batch_size, remaining_episodes)
                self.env.load_problems(batch_size=batch_size,
                                       problem_size=problem_size,
                                       validation_data=dataset,
                                       device=self.device,
                                       problem_name=problem_name,
                                       start=tested_episodes)
                reset_state, _, _ = self.env.reset()

                self.model.decoder.assign(problem_representation)
                self.model.pre_forward(reset_state, problem_name, problem_representation)

                # POMO Rollout
                ###############################################
                state, reward, done = self.env.pre_step()
                while not done:
                    cur_dist = self.env.get_local_feature()
                    selected, _ = self.model(state, cur_dist)
                    # shape: (batch, pomo)
                    state, reward, done = self.env.step(selected)

                # Return
                ###############################################
                max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
                # shape: (batch,)
                results[tested_episodes:tested_episodes + batch_size] = max_pomo_reward
                tested_episodes += batch_size
        if problem_name in ["op"]:
            avg_score = results.mean()
            gap = ((optimal_score - avg_score) * 100 / optimal_score).item()
        else:
            avg_score = -results.mean() # negative sign to make positive value
            gap = ((avg_score - optimal_score) * 100 / optimal_score).item()
        avg_score_item = avg_score.item()
        # Logs
        ##################################################
        self.result_log.append(f'{problem_name}_eval_{problem_size}', epoch, avg_score_item)
        self.result_log.append(f'{problem_name}_gap_{problem_size}', epoch, gap)


        self.logger.info('Epoch {:4d}: In {:7d} {:4d}-nodes instances, Problem name: {:4s},  Score: {:.4f}, Gap: {:.4f}%'.format(
                epoch, tested_episodes,problem_size, problem_name, avg_score_item, gap))

        if epoch > 1:
            image_prefix = '{}/latest'.format(self.result_folder)
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                           self.result_log, labels=[f'{problem_name}_eval_{problem_size}'])
            util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                           self.result_log, labels=[f'{problem_name}_gap_{problem_size}'])

        # Update best gap
        if gap < self.problem_gaps[problem_name][problem_size]:
            self.problem_gaps[problem_name][problem_size] = gap
            self.logger.info('Epoch {:4d}: New best gap for {:4s} {:4d}-nodes: {:.4f}%'.format(
                epoch, problem_name, problem_size, gap))

        return avg_score, gap

