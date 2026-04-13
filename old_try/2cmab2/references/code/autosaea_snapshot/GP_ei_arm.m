% =========================================================================
% GP_ei_arm: 组合臂 {GP, EI} 的执行函数
% 对应论文 IV.A节 第1)点 + 公式(2)-(6)(13)
%
% 工作流程:
%   1. 用当前种群训练GP模型
%   2. 对每个DE子代, 用EI准则评价
%   3. 选EI值最大的子代作为候选解
%   4. 真实评估 + 计算奖励
%
% EI(Expected Improvement)的直觉:
%   EI 衡量新点 x "期望能比当前最优值改进多少"
%   EI(x) = (f_min - f̂(x))·Φ(Z) + ŝ(x)·φ(Z)
%   其中 Z = (f_min - f̂(x)) / ŝ(x)
%   - 第一项: 如果预测值远好于当前最优, 贡献大 (利用)
%   - 第二项: 如果不确定性大, 贡献大 (探索)
%   EI 自然地平衡了探索和利用, 是贝叶斯优化中最经典的获取函数
%
% 论文公式(13):
%   x* = argmax_{x∈X} [ (f_min - f̂_{GP}(x))·Φ(Z) + ŝ(x)·φ(Z) ]
%   其中 Z = (f_min - f̂_{GP}(x)) / ŝ(x)
%   f_min = 当前最优目标函数值
%   Φ(·) = 标准正态分布的累积分布函数(CDF)
%   φ(·) = 标准正态分布的概率密度函数(PDF)
% =========================================================================
function [hx, hf, reward, NFEs, CE, gfs] = GP_ei_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs)

try  % try-catch: GP训练可能因数值问题失败
     theta =[];                             % GP超参数(自动优化)

     %% --- 数据预处理: 去除重复样本 ---
     t_s = [ghx, ghf'];
     D = size(ghx, 2);
     [t_s, ~,~] = unique(t_s, 'rows');      % 去重(避免协方差矩阵奇异)
     ghx = t_s(:, 1:D);
     ghf = t_s(:, D+1);

     %% --- 步骤1: 训练GP模型 ---
     % dacefit_3 是DACE工具箱的GP训练函数(与GP_fit类似)
     try
        [dmodel, ~]=...
         dacefit_3(ghx,ghf,@regpoly0,@corrgauss,theta);
     catch
        [dmodel, ~]=...
        dacefit_3(ghx,ghf,@regpoly0,@corrgauss,theta);
     end

     %% --- 步骤2+3: 计算每个子代的EI值 ---
     Gbest = min(hf);                       % f_min: 当前全局最优目标函数值
     for i = 1 : size(offspring, 1)          % 遍历每个DE子代
         % predictor 返回:
         %   y: 预测均值 f̂_{GP}(x) (公式5)
         %   mse: 预测均方误差 ŝ²(x) (公式6)
         [y,~, mse, ~] =  predictor(offspring(i, :),dmodel);
         s = sqrt(mse);                     % ŝ(x): 预测标准差

         % --- 计算EI, 对应论文公式(13) ---
         % EI = (f_min - ŷ)·Φ((f_min - ŷ)/s) + s·φ((f_min - ŷ)/s)
         % 代码中取负号(因为后面用min选择): -EI
         % normcdf: 标准正态CDF Φ(·)
         % normpdf: 标准正态PDF φ(·)
         EI(i) = -(Gbest-y)*normcdf((Gbest-y)/s)-s*normpdf((Gbest-y)/s);
     end

     %% --- 选EI最大的(即-EI最小的) ---
     % 对应论文公式(13): x* = argmax EI(x) → 等价于 argmin(-EI)
     [~,I] = min(EI);
     candidate_position = offspring(I, :);    % 候选解 x_t

     %% --- 去重检查 + 真实评估 + 奖励计算 ---
     [~,ih,~] = intersect(hx,candidate_position,'rows');
     if isempty(ih)==1
        candidate_fit=FUN(candidate_position);  % 真实评估
        NFEs = NFEs + 1;
        hx=[hx; candidate_position];  hf=[hf, candidate_fit];

        CE(NFEs,:)=[NFEs,candidate_fit];
        gfs(1,NFEs)=min(CE(1:NFEs,2));

        Arm = num2str('GP_ei ');
        [reward] = Low_level_r(ghf, hf, candidate_fit, NFEs, Arm);  % 低层奖励(公式24)
     else
         reward = 0;
     end

catch
     reward = 0;    % GP训练失败, 奖励为0
end

end
