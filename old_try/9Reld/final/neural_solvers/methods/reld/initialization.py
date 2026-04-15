from EasyNCO.neural_solvers.pipeline import ARInitialization
from typing import Any, Tuple
from tensordict import TensorDict
import torch

class ReLDInitialization(ARInitialization):
    """
    ReLD初始化类 - 基于POMO初始化，但适配ReLD的增强解码器
    处理CVRP问题的特殊需求，如容量约束和距离计算
    """
    def play_episode(
        self,
        env,
        decoder_strategy: str = "sampling",
    ) -> Tuple[TensorDict, dict]:
        """
        播放一个完整的环境episode
        为ReLD解码器提供必要的信息，如当前距离等
        """
        reset_td = env.reset()

        # 设置解码策略
        self.policy.set_decoder_strategy(decoder_strategy)
        self.policy.pre_forward(reset_td)

        # 初始化likelihood tensor
        likelihood = torch.zeros(size=(env.batch_size[0], env.pomo_size, 0))

        done = False
        reward = None
        state_td = env.pre_step()
        
        # 为ReLD解码器获取初始距离信息
        if hasattr(env, '_get_local_feature') and callable(env._get_local_feature):
            local_feature = env._get_local_feature()
            
            if local_feature is None:
                cur_dist = None
            elif isinstance(local_feature, tuple):
                # 当 current_node is None 时返回的 tuple 格式
                cur_dist = local_feature[0] if len(local_feature) > 0 else None
            elif isinstance(local_feature, dict):
                # 正常情况下的字典格式
                cur_dist = local_feature.get('cur_dist', None)
            else:
                cur_dist = None
        else:
            cur_dist = None

        # 主推理循环
        while not done:
            # 为ReLD解码器传递当前距离信息
            if cur_dist is not None:
                next_td = self.policy(state_td, cur_dist=cur_dist)
            else:
                next_td = self.policy(state_td)
                
            prob = next_td.get("prob", None)
            
            # 环境步进
            state_td = env.step(next_td)
            likelihood = torch.cat((likelihood, prob[:, :, None]), dim=2)

            # 更新距离信息
            if hasattr(env, '_get_local_feature') and callable(env._get_local_feature):
                local_feature = env._get_local_feature()
                
                if local_feature is None:
                    cur_dist = None
                elif isinstance(local_feature, tuple):
                    # 当 current_node is None 时返回的 tuple 格式
                    cur_dist = local_feature[0] if len(local_feature) > 0 else None
                elif isinstance(local_feature, dict):
                    # 正常情况下的字典格式
                    cur_dist = local_feature.get('cur_dist', None)
                else:
                    cur_dist = None

            reward = state_td.get("reward", None)
            done = state_td.get("done", torch.tensor(False)).all()

        policy_out = {
            "reward": reward,
            "likelihood": likelihood,
        }
        return state_td, policy_out