from neural_solvers.pipeline.initialization import Initialization
from typing import Any, Tuple, Literal
from torchrl.envs import EnvBase
from tensordict import TensorDict
import torch
import copy
from EasyNCO.neural_solvers.methods.omni.utils import generate_task_set,_update_task_weight
from EasyNCO.neural_solvers.methods.omni.train_utils import get_data,manual_adam_omni,_fast_val
from torch.optim import Adam as Optimizer
from collections import OrderedDict

class OMNIInitialization(Initialization):
    def set_parameter(self,**kwargs):
        meta_params = kwargs.get('meta_params')
        self.so_to_fo = True if meta_params['meta_method'] == 'maml_fomaml' else False
        self.task_set = generate_task_set(meta_params["data_type"])
        self.val_data, self.val_opt = {}, {}  # for lkh3_offline
        if meta_params["data_type"] == "size_distribution":
            self.min_n, self.max_n, self.task_interval, self.num_dist = 50, 200, 5, 11
            self.task_w = torch.full(((self.max_n - self.min_n) // self.task_interval + 1, self.num_dist),
                                     1 / self.num_dist)



    def run(self,
            env: EnvBase,
            batch: int,  #omni_train do not use this batch
            strategy: str,
            phase: Literal["train", "eval"],
            **kwargs,
    ) -> Tuple[TensorDict, Any]:
        if phase == "train":
            meta_params = kwargs.get('meta_params', {})
            current_epoch = kwargs.get('current_epoch')
            
            meta_optimizer = self.optimizers()
            meta_optimizer.zero_grad()
            if self.so_to_fo:
                if current_epoch > meta_params['switch_epochs']:
                    meta_params['meta_method'] = 'fomaml'
                else:
                    meta_params['meta_method'] = 'maml'
            # Adaptive task scheduler:
            if meta_params["data_type"] == "size_distribution":
                # omni 默认
                start = self.min_n + int(
                    min(current_epoch / meta_params['sch_epoch'], 1) * (self.max_n - self.min_n))  # linear
                n = start // self.task_interval * self.task_interval
                idx = (n - self.min_n) // self.task_interval
                tasks, weights = self.task_set[idx * 11: (idx + 1) * 11], self.task_w[idx]
                if current_epoch +1  % meta_params['update_weight'] == 0:
                    self.task_w[idx] = _update_task_weight(self, tasks, weights,meta_params,env)

            val_loss, meta_grad_dict = 0, {(i, j): 0 for i, group in
                                           enumerate(meta_optimizer.param_groups) for j, _ in
                                           enumerate(group['params'])}
            # sample a batch of tasks
            meta_batch_size = meta_params['meta_batch_size']
            # w is for task select
            w, selected_tasks = [1.0] * meta_params['B'], []
            for b in range(meta_params['B']):
                if meta_params["data_type"] == "size_distribution":
                    selected = torch.multinomial(self.task_w[idx], 1).item()
                    task_params = tasks[selected]
                    w[b] = self.task_w[idx][selected].item()
                    batch_size = meta_batch_size if task_params[0] <= 150 else meta_batch_size // 2
                selected_tasks.append(task_params)
            w = torch.softmax(torch.Tensor(w), dim=0)

            # outer-loop optimization
            for b in range(meta_params['B']):
                task_params, task_w = selected_tasks[b], w[b].item()
                # preparation
                if meta_params['meta_method'] == 'fomaml':
                    task_model = copy.deepcopy(self.policy)
                    optimizer = Optimizer(task_model.parameters(), **self.optimizer_params['optimizer'])
                elif meta_params['meta_method'] == 'maml':
                    fast_weight = OrderedDict(self.policy.named_parameters())

                # inner-loop optimization
                for step in range(meta_params['k']):
                    data,norm_data = get_data(meta_params['data_type'], batch_size, task_params, env.problem_size,
                                    env.env_name)
                    if env.env_name == 'tsp':
                        env.pomo_size = norm_data.size(1)
                        env.problem_size = norm_data.size(1)
                    else:
                        env.pomo_size = norm_data.size(1) - 1
                        env.problem_size = norm_data.size(1) - 1
                    env.load_problems(norm_data, batch_size)
                    self.policy.train()
                    if meta_params['meta_method'] == 'fomaml':
                        task_model.train()
                        state_td, policy_out = self.play_episode(env, 'sampling', policy=task_model)
                        loss = self.calculate_loss(policy_out)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    elif meta_params['meta_method'] == 'maml':
                        state_td, policy_out = self.play_episode(env, 'sampling',
                                                                 weights=fast_weight)
                        loss = self.calculate_loss(policy_out)
                        fast_weight = manual_adam_omni(self=self, loss=loss, fast_weight=fast_weight,
                                                       meta_optimizer=meta_optimizer)

                    reward = policy_out["reward"]


                val_data,norm_val_data = get_data(meta_params['data_type'], batch_size, task_params, env.problem_size,
                                    env.env_name)

                if meta_params['meta_method'] == 'maml':
                    val_loss = _fast_val(self, env, fast_weight, data=norm_val_data, mode="maml")
                    # logger.info(val_loss)
                    loss = meta_params['beta'] * val_loss * task_w
                    meta_optimizer.zero_grad()
                    loss.backward()
                    for i, group in enumerate(meta_optimizer.param_groups):
                        for j, p in enumerate(group['params']):
                            if p.grad is None:
                                continue
                            else:
                                meta_grad_dict[(i, j)] += p.grad

                elif meta_params['meta_method'] == 'fomaml':
                    val_loss = _fast_val(self,env,task_model, data=val_data, mode="fomaml")
                    # logger.info(val_lss)
                    loss = meta_params['beta'] * val_loss * task_w
                    optimizer.zero_grad()
                    loss.backward()
                    for i, group in enumerate(optimizer.param_groups):
                        for j, p in enumerate(group['params']):
                            if p.grad is None:
                                continue
                            else:
                                meta_grad_dict[(i, j)] += p.grad



            # outer-loop optimization (update meta-model)
            meta_optimizer.zero_grad()
            for i, group in enumerate(meta_optimizer.param_groups):
                for j, p in enumerate(group['params']):
                    if isinstance(meta_grad_dict[(i, j)], int):
                        continue
                    else:
                        p.grad = meta_grad_dict[(i, j)]
            meta_optimizer.step()
            ret = {}
            ret['batch_size'] = batch_size
            ret['reward'] = reward #inner_batch_reward
            ret['loss'] = loss   #meta_loss
            return ret, ret

        else:
            self.policy.eval()
            with torch.inference_mode():
                env.load_problems(batch, batch_size=batch.size(0))
                # if env.env_name =='cvrp':
                #     env.problem_size = batch.size(1)-1
                state_td, policy_out = self.play_episode(env, strategy)
            aug_reward = policy_out["reward"].reshape(
                env.aug_factor, batch.size(0), env.pomo_size
            )
            # shape: (augmentation, batch, pomo)

            max_pomo_reward, _ = aug_reward.max(dim=2)  # best result from pomo
            # shape: (augmentation, batch)

            no_aug_reward = (
                -max_pomo_reward[0, :].float().mean()
            )  # negative sign to make positive value

            max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best result from augmentation
            # shape (batch, )
            aug_score = (
                -max_aug_pomo_reward.float().mean()
            )  # negative sign to make positive value

            out = {
                "no_aug_score": no_aug_reward,
                "aug_score": aug_score,
            }

            fast_val_flag =kwargs.get('fast_val_flag')
            if fast_val_flag is not None:
                return no_aug_reward,aug_score,max_pomo_reward

            return state_td, out


    def play_episode(
            self,
            env,
            decoder_strategy: str = "sampling",
            weights = None,
            policy = None,
    ) -> Tuple[TensorDict, dict]:
        if policy is None:
            policy = self.policy
        reset_td = env.reset()
        policy.set_decoder_strategy(decoder_strategy)
        if weights is None:
            policy.pre_forward(reset_td)
        else:
            policy.pre_forward(reset_td, weights=weights)

        log_likelihood = torch.zeros(size=(env.batch_size[0], env.pomo_size, 0))
        done = False
        reward = None
        state_td = env.pre_step()

        while not done:
            if weights is None:
                next_td = policy(state_td)
            else:
                next_td = policy(state_td,weights=weights)
            prob = next_td["prob"]
            state_td = env.step(next_td)
            log_likelihood = torch.cat((log_likelihood, prob[:, :, None]), dim=2)
            reward = state_td["reward"]
            done = state_td["done"].all()

        policy_out = {
            "reward": reward,
            "likelihood": log_likelihood,
        }
        return state_td, policy_out