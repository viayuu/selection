import torch
import math
from collections import OrderedDict



def manual_adam_omni(self,loss,fast_weight,meta_optimizer):

    gradients = torch.autograd.grad(loss, fast_weight.values(),
                                    create_graph=True,allow_unused=True)  # allow_unused=True
    w_t, (beta1, beta2), eps = [], meta_optimizer.param_groups[0]['betas'], \
        meta_optimizer.param_groups[0]['eps']
    lr, weight_decay = self.optimizer_params['optimizer']['lr'], self.optimizer_params['optimizer'][
        'weight_decay']
    for i, ((name, param), grad) in enumerate(zip(fast_weight.items(), gradients)):
        if name == 'decoder.GraphMeanEmbedding.graph_mean_embedding.weight':  # POMO does not have this
            continue
        if meta_optimizer.state_dict()['state'] != {}:
            # (with batch/instnace norm layer): i \in [0, 85], where encoder \in [0, 79] + decoder \in [80, 85]
            # (with rezero norm layer): i \in [0, 73], where encoder \in [0, 67] + decoder \in [68, 73]
            # (without norm layer): i \in [0, 61], where encoder \in [0, 55] + decoder \in [56, 61]

            state = meta_optimizer.state_dict()['state'][i]
            step, exp_avg, exp_avg_sq = state['step'], state['exp_avg'], state['exp_avg_sq']
            step += 1
            step = step.item() if isinstance(step, torch.Tensor) else step
            # compute grad based on Adam source code using in-place operation
            # update Adam stat (step, exp_avg and exp_avg_sq have already been updated by in-place operation)
            # may encounter RuntimeError: (a leaf Variable that requires grad) / (the tensor used during grad computation) cannot use in-place operation.
            grad = grad.add(param, alpha=weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            # param.addcdiv_(exp_avg, denom, value=-step_size)
            param = param - step_size * exp_avg / denom
            meta_optimizer.state_dict()['state'][i]['exp_avg'] = exp_avg.clone().detach()
            meta_optimizer.state_dict()['state'][i]['exp_avg_sq'] = exp_avg_sq.clone().detach()
        else:
            param = param - lr * grad
        w_t.append((name, param))
    fast_weight = OrderedDict(w_t)

    return fast_weight

def get_data(data_type, batch_size, task_params,problem_size,env_name):
    #avoid circular import
    from EasyNCO.neural_solvers.methods.omni.utils import _get_data
    return _get_data(data_type, batch_size, task_params,problem_size,env_name)


def _fast_val(self, env, model, data=None, mode="eval", return_all=False):
    origin_aug_factor = env.aug_factor
    env.aug_factor = 1


    batch_size = data.size(0)
    if env.env_name == 'tsp':
        env.pomo_size = data.size(1)
        env.problem_size = data.size(1)
    else:
        env.pomo_size = data.size(1)-1
        env.problem_size = data.size(1)-1

    if mode == "eval":
        model.eval()
        with torch.no_grad():
            no_aug_reward, aug_score, max_pomo_reward = self.run(env,data,'greedy','eval',fast_val_flag=True)

    elif mode in ["maml", "fomaml"]:
        env.load_problems(dataset=data, batch_size=batch_size)
        fast_weight = model
        if mode == "maml":
            state_td, policy_out = self.play_episode(env, 'sampling', weights=fast_weight)
            loss = self.calculate_loss(policy_out)
        else:
            state_td, policy_out = self.play_episode(env, 'sampling',policy=model)
            loss = self.calculate_loss(policy_out)

    else:
        raise NotImplementedError

    env.aug_factor = origin_aug_factor


    if mode == "eval":
        if return_all:
            return max_pomo_reward
        else:
            return no_aug_reward.detach().item()
    else:
        return loss


