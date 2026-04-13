% =========================================================================
% TL-UCB: 两层上置信界策略 (Two-Level Upper Confidence Bound)
% 对应论文 Algorithm 2, 以及论文公式(22)和(23)
%
% UCB的核心思想(多臂老虎机理论):
%   UCB值 = 利用项(exploitation) + 探索项(exploration)
%   - 利用项 q_value: 该摇臂的平均奖励, 越高说明历史表现越好
%   - 探索项 sqrt(α·ln(t)/T(a)): 被选次数少的摇臂会有更大的探索奖励
%   选择UCB值最大的摇臂, 自然平衡了"选历史最好的"和"尝试未知的"
%
% 输入:
%   sum_reward - 该摇臂的所有历史奖励向量, 其长度即为该摇臂被选择的次数 T_a(t)
%   id         - 当前迭代轮次 t (对应论文公式中的 t)
%   q_value    - 该摇臂的当前Q值(平均奖励), 对应论文公式(22)中的 Q_a^H(t)
%                或公式(23)中的 Q_a^L(t)
%   apha       - 探索参数α (论文中设为2.5), 控制探索-利用的平衡程度
%                α越大, 越鼓励探索较少使用的摇臂
%
% 输出:
%   U_value    - 该摇臂的UCB值, 用于与其他摇臂比较, 选UCB值最大的
%
% 论文公式(22) [高层]:
%   a_t^H = argmax_{a∈A^H} [ Q_a^H(t) + sqrt( α·ln(t) / T_a^H(t) ) ]
%
% 论文公式(23) [低层]:
%   a_t^L = argmax_{a∈A_{a_t^H}^L} [ Q_a^L(t) + sqrt( α·ln(t) / T_a^L(t) ) ]
%
% 高层和低层使用完全相同的UCB公式, 只是输入的Q值和T值不同
% =========================================================================
function [U_value] = TL_UCB(sum_reward, id, q_value, apha)
    % q_value: Q_a(t), 该摇臂的平均奖励 (利用项, 反映历史表现)
    % sqrt(α·ln(t) / T_a(t)): 探索项
    %   - apha * log(id): α·ln(t), 随总轮次t增长而缓慢增大
    %   - length(sum_reward): T_a(t), 该摇臂被选择的总次数
    %   - 当T_a(t)很小(很少被选)时, 探索项很大, 鼓励尝试
    %   - 当T_a(t)很大(经常被选)时, 探索项很小, 主要看利用项
    U_value = q_value + sqrt((apha*log(id))/length(sum_reward));
end
