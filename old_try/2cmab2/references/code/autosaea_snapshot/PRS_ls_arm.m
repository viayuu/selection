% =========================================================================
% PRS_ls_arm: 组合臂 {PRS, local search} 的执行函数
% 对应论文 IV.A节 第2)点 + 公式(9)(10)(17)(18)
%
% 工作流程:
%   1. 用当前种群训练PRS代理模型
%   2. 在当前种群的包围盒内, 用DE优化PRS模型 (local search)
%   3. 真实评估DE找到的PRS模型最优解
%   4. 计算低层奖励
%
% 与 PRS prescreening 的区别:
%   - prescreening: 从N个子代中选最好的(快但搜索范围有限)
%   - local search: 在代理模型上跑完整的DE优化(慢但搜索更深入)
% =========================================================================
function [hx, hf, reward,  NFEs,  CE, gfs] = PRS_ls_arm(ghx, ghf, hx, hf, FUN, NFEs, CE, gfs)

    flag = 0;                               % flag=0 表示使用PRS模型(DE_optimizer中的分支)
    a1 = 100;
    Dim = size(ghx, 2);                     % 决策变量维度 D
    Max_NFE = a1*Dim+1000;                  % DE在PRS模型上的最大评估次数 = 100D+1000
    minerror = 1e-20;                       % 收敛阈值

    %% --- 步骤1: 训练PRS代理模型 ---
    srgtOPT=srgtsPRSSetOptions(ghx, ghf');
    srgtSRGT = srgtsPRSFit(srgtOPT);

    %% --- 步骤2: 在PRS模型上进行DE局部搜索 ---
    % DE_optimizer 内部搜索范围 = [min(ghx), max(ghx)] (论文公式(18))
    % flag=0 告诉DE_optimizer使用PRS模型进行预测
    [candidate_position,~, ~] = DE_optimizer(Dim, Max_NFE,srgtSRGT,minerror,ghx, flag);

    %% --- 去重检查 + 真实评估 + 奖励计算 ---
    [~,ih,~] = intersect(hx,candidate_position,'rows');
    if isempty(ih)==1
        candidate_fit=FUN(candidate_position);  % 真实评估
        NFEs = NFEs + 1;
        hx=[hx; candidate_position];  hf=[hf, candidate_fit];

        CE(NFEs,:)=[NFEs,candidate_fit];
        gfs(1,NFEs)=min(CE(1:NFEs,2));

        Arm = num2str('PRS_local_search ');
        [reward] = Low_level_r(ghf, hf, candidate_fit, NFEs, Arm);
    else
        reward = 0;
    end

end
