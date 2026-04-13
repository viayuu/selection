% =========================================================================
% PRS_pre_arm: 组合臂 {PRS, prescreening} 的执行函数
% 对应论文 IV.A节 第2)点 + 公式(9)(10)(17)
%
% 工作流程:
%   1. 用当前种群训练PRS(二阶多项式响应面)代理模型 (论文公式(9)(10))
%   2. 用PRS模型预测所有DE子代的适应度
%   3. 选预测最优的子代作为候选解 (prescreening, 论文公式(17))
%   4. 真实评估 + 计算奖励
%
% PRS模型 (论文公式(9)):
%   f̂_{PRS}(x) = β₀ + Σ βᵢxᵢ + Σ βᵢⱼxᵢxⱼ
%   系数通过最小二乘法求解 (论文公式(10)): β = (P'P)⁻¹P'F
% =========================================================================
function [hx, hf, reward,  NFEs,  CE, gfs] = PRS_pre_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs)

    %% --- 步骤1: 训练PRS代理模型 ---
    % PRS = Polynomial Response Surface (多项式响应面)
    % 使用二阶多项式拟合, 对应论文公式(9)
    srgtOPT=srgtsPRSSetOptions(ghx, ghf');   % 设置PRS训练选项
    srgtSRGT = srgtsPRSFit(srgtOPT);         % 求解系数 β (论文公式(10))

    %% --- 步骤2: 用PRS模型预测所有子代的适应度 ---
    fitnessModel = srgtsPRSEvaluate(offspring, srgtSRGT);  % f̂_{PRS}(offspring_i)

    %% --- 步骤3: Prescreening —— 选预测最优的子代 ---
    % 对应论文公式(17): x* = argmin_{x∈X} f̂_{PRS}(x)
    [~,sidx]=min(fitnessModel);              % 找预测最优的索引
    candidate_position = offspring(sidx, :); % 候选解 x_t

    %% --- 去重检查 ---
    [~,ih,~] = intersect(hx,candidate_position,'rows');

    if isempty(ih)==1                        % 候选解是新的
        candidate_fit=FUN(candidate_position);  % 真实评估
        NFEs = NFEs + 1;
        hx=[hx; candidate_position];  hf=[hf, candidate_fit];  % 更新数据库

        CE(NFEs,:)=[NFEs,candidate_fit];
        gfs(1,NFEs)=min(CE(1:NFEs,2));

        Arm = num2str('PRS_prescreening');
        [reward] = Low_level_r(ghf, hf, candidate_fit, NFEs, Arm);  % 低层奖励(公式24)
    else
        reward = 0;                          % 重复解, 奖励为0
    end

end
