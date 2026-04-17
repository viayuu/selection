import os
import random

import numpy as np
import torch
from logging import getLogger


from UNIEnv import UniEnv
from Model import Model as Model
from multi_hot_set import *
from 参考文献.URS代码.utils.utils import *
from ProblemDef import get_saved_data
from 参考文献.URS代码.mask.mask_registry import mask_registry


class Tester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,
                 test_problem_list=None):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.test_problem_list = test_problem_list

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        self.problem_attributes_set = get_problem_attributes_set()
        self.logger.info("Testing problem list(num: {}): {}".format(len(self.test_problem_list), self.test_problem_list))

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # Main Components
        self.model = Model(**self.model_params).to(self.device)

        # Restore
        model_load = tester_params['model_load']
        if "epoch" in model_load.keys():
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        else:
            checkpoint_fullname = '{path}/{name}.pt'.format(**model_load)

        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info("Model loaded successfully!!!")
        self.logger.info("Model loaded from: {0}".format(checkpoint_fullname))

        total = sum([param.nelement() for param in self.model.parameters()])
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

        # utility
        self.time_estimator = TimeEstimator()

        # 1. 创建一个包含所有问题配置的列表
        # 每个字典代表一个问题的配置信息
        self.problem_dataset = get_problem_validation_set()
        self.saved_data = {}
        self.optimal_scores = {}
        validation_scale_list = self.env_params['problem_size_list']
        for scale in validation_scale_list:
            # 2. 遍历问题配置列表，加载数据
            self.saved_data[scale] = {}
            self.optimal_scores[scale] = {}
            for problem_name in self.test_problem_list:

                detailed_validation = self.problem_dataset[problem_name]

                num_instances = detailed_validation.get(f"episodes_{scale}", 1000)  # 默认10000个实例

                filename = os.path.join("../data", problem_name, detailed_validation.get(f"filename_{scale}", ""))
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

                self.logger.info('Successfully load {0:5d} {1:20s} instances, optimal_score: {2:7.4f}'.format(
                    num_instances, problem_name.upper()+str(scale), optimal_score))

    def run(self):

        for problem_size in self.env_params['problem_size_list']:
            gap_log_parts = []
            aug_gap_log_parts = []
            self.problem_gaps = {name: {'gap': 100.0, 'aug_gap': 100.0} for name in self.test_problem_list}
            for problem_name in self.test_problem_list:
                self.logger.info('========================================================================')
                self.logger.info('========================================================================')

                #########################################################################
                # Inference
                #########################################################################
                test_num_episode = self.problem_dataset[problem_name][f'episodes_{problem_size}']
                test_num_episode = min(test_num_episode, self.tester_params['test_episodes'])
                saved_dataset = self.saved_data[problem_size][problem_name]
                optimal_score = self.optimal_scores[problem_size][problem_name]

                # set augmentation factor and batch size
                if problem_size > 1000:
                    batch_size_list =  {2000:8, 3000:2, 4000:1, 5000:1}
                    self.logger.info(f"for problem size > 1000, we use batch_size_list{batch_size_list}")
                    batch_size = batch_size_list[problem_size]
                    aug_factor = self.tester_params['aug_factor']
                else:
                    if self.tester_params['augmentation_enable']:
                        if 'a' in problem_name:
                            # asymmetric problems
                            aug_factor = 128
                            if 'md' in problem_name:
                                batch_size = 8
                            else:
                                batch_size = 50
                        else:
                            # symmetric problems
                            aug_factor = 8
                            if 'md' in problem_name:
                                batch_size = 125
                            else:
                                batch_size = self.tester_params['test_batch_size']
                    else:
                        aug_factor = 1
                        batch_size = self.tester_params['test_batch_size']

                self.logger.info(f" Testing {problem_name} {problem_size} with aug_factor: {aug_factor}, batch_size: {batch_size}")

                self.test_one_problem(problem_name, saved_dataset, optimal_score,
                                           test_num_episode, problem_size,aug_factor=aug_factor,
                                           batch_size=batch_size)
                gap = self.problem_gaps[problem_name]['gap']
                aug_gap = self.problem_gaps[problem_name]['aug_gap']
                gap_log_parts.append(f" {problem_name.upper()}{problem_size}: {gap:.4f}%")
                aug_gap_log_parts.append(f" {problem_name.upper()}{problem_size}: {aug_gap:.4f}%")

            self.logger.info('===============================Test Summary===================================')
            gap_log_str = f"scale: {problem_size} ==> Current detailed gaps (no aug): " + ", ".join(gap_log_parts)
            self.logger.info(gap_log_str)
            aug_gap_log_str = f"scale: {problem_size} ==> Current detailed gaps (aug): " + ", ".join(aug_gap_log_parts)
            self.logger.info(aug_gap_log_str)


    def test_one_problem(self, problem_name, dataset, optimal_score,episodes,problem_size,aug_factor=1,batch_size=None):

        self.model.eval()
        self.model.set_decoder_type("greedy")
        self.env = UniEnv()

        # Get mask function from registry
        mask_fn = mask_registry.get_mask_combo(problem_name)

        self.time_estimator.reset()
        score = AverageMeter()
        aug_score = AverageMeter()
        self.logger.info(f" Reset time estimator and averageMeter for {problem_name} {problem_size} inference")

        problem_representation = torch.tensor(self.problem_attributes_set[problem_name], dtype=torch.float32,device= self.device)
        tested_episodes = 0
        if batch_size is None:
            batch_size = episodes

        with torch.inference_mode():
            start_time = time.time()
            while tested_episodes < episodes:
                remaining_episodes = episodes - tested_episodes
                batch_size = min(batch_size, remaining_episodes)
                self.env.load_problems(batch_size=batch_size,
                                       problem_size=problem_size,
                                       pomo_size=problem_size,
                                       validation_data=dataset,
                                       aug_factor=aug_factor,
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
                    state, reward, done = self.env.step(selected,mask_fn=mask_fn)


                # Return
                ###############################################
                aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
                # shape: (augmentation, batch, pomo)

                max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
                # shape: (augmentation, batch)
                if problem_name in ["op"]:
                    avg_score = max_pomo_reward[0, :].float().mean().item()
                else:
                    avg_score = -max_pomo_reward[0, :].float().mean().item()  # negative sign to make positive value

                max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
                # shape: (batch,)
                if problem_name in ["op"]:
                    avg_aug_score = max_aug_pomo_reward.float().mean().item()
                else:
                    avg_aug_score = -max_aug_pomo_reward.float().mean().item()  # negative sign to make positive value
                    # shape: (batch,)

                score.update(avg_score, batch_size)
                aug_score.update(avg_aug_score, batch_size)

                tested_episodes += batch_size

                ############################
                # Logs
                ############################
                elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(tested_episodes, episodes)
                self.logger.info(" episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                        tested_episodes, episodes, elapsed_time_str, remain_time_str, avg_score, avg_aug_score))

                all_done = (tested_episodes == episodes)
                if all_done:
                    end_time = time.time()
                    if problem_name in ["op"]:
                        gap = ((optimal_score - score.avg) * 100 / optimal_score)
                        aug_gap = ((optimal_score - aug_score.avg) * 100 / optimal_score)
                    else:
                        gap = ((score.avg - optimal_score) * 100 / optimal_score)
                        aug_gap = ((aug_score.avg - optimal_score) * 100 / optimal_score)
                    self.logger.info(f" ************* Test Done: {problem_name} ************* ")
                    self.logger.info(" problem size: {0}, pomo size: {1}, optimal score: {2:.4f} ".format(
                            problem_size, self.env.pomo_size, optimal_score))
                    self.logger.info(" NO-AUG SCORE:{0:.4f}, GAP:{1:.3f}%".format(
                        score.avg, gap))
                    self.logger.info(" AUGMENTATION SCORE:{0:.4f}, GAP:{1:.3f}%".format(
                        aug_score.avg, aug_gap))

                    self.logger.info(" Total time: {:.2f} sec".format(end_time - start_time))
                    self.logger.info(" Avg time per episode: {:.2f} sec".format((end_time - start_time) / episodes))

                    # Update best gap
                    self.problem_gaps[problem_name]['gap'] = gap
                    self.problem_gaps[problem_name]['aug_gap'] = aug_gap
