% =========================================================================
% RBF_ls_arm: 组合臂 {RBF, local search} 的执行函数
% 对应论文 IV.A节 第2)点 + 公式(7)(8)(17)(18)
%
% 工作流程:
%   1. 用当前种群训练RBF代理模型
%   2. 在当前种群的包围盒内, 用DE优化RBF模型 (local search)
%   3. DE找到的RBF模型最优解作为候选解
%   4. 对候选解进行真实评估
%   5. 计算低层奖励
%
% 与 prescreening 的区别:
%   - prescreening: 只从DE生成的N个子代中选最好的(有限集合搜索)
%   - local search: 在代理模型上运行完整的DE优化(连续空间搜索)
%   local search 更深入但计算量更大(最多 100D+1000 代 × N 次代理模型评估)
%
% 论文公式(17)+(18) [local search]:
%   x* = argmin_{x∈X} f̂_{RBF}(x), 其中 X = [lb, ub]
%   lb_j = min{x_{i,j}, x_i ∈ P}, ub_j = max{x_{i,j}, x_i ∈ P}
%   "如果X是搜索空间的子空间, 公式(17)属于local search"
% =========================================================================
function [hx, hf, reward,  NFEs,  CE, gfs] = RBF_ls_arm(ghx, ghf, hx, hf, FUN, NFEs, CE, gfs)

    %% --- 设置local search参数 ---
    a1 = 100;
    Dim = size(hx, 2);                      % 决策变量维度 D
    Max_NFE = a1*Dim+1000;                  % DE在代理模型上的最大评估次数 = 100D+1000
                                            % 论文: "DE local search is conducted with
                                            % a maximum number of 100D+1000 generations"
    minerror = 1e-20;                       % 收敛阈值(连续10代改进小于此值则停止)

    %% --- 步骤1: 训练RBF代理模型 ---
    % 同 RBF_pre_arm, 使用立方基函数(CUB)
    srgtOPT=srgtsRBFSetOptions(ghx, ghf', @my_rbfbuild, [],'CUB', 0.0002,1);
    srgtSRGT = srgtsRBFFit(srgtOPT);

    %% --- 步骤2: 在代理模型上进行DE局部搜索 ---
    % flag=1 表示使用RBF模型
    % DE_optimizer 会在种群包围盒[min(ghx), max(ghx)]内搜索RBF模型的最优解
    % 注意: 这里的评估都在代理模型上, 不消耗真实FEs
    flag = 1;
    [candidate_position,~, ~] = DE_optimizer(Dim,Max_NFE,srgtSRGT,minerror,ghx, flag);

    %% --- 去重检查 ---
    [~,ih,~] = intersect(hx,candidate_position,'rows');

    if isempty(ih)==1       % 候选解是新的
        %% --- 真实评估 + 更新数据库 + 计算奖励 ---
        candidate_fit=FUN(candidate_position);  % 真实评估 (消耗1次FE)
        NFEs = NFEs + 1;

        hx=[hx; candidate_position];  hf=[hf, candidate_fit];  % 更新数据库

        CE(NFEs,:)=[NFEs,candidate_fit];
        gfs(1,NFEs)=min(CE(1:NFEs,2));

        Arm = num2str('RBF_local_search ');
        [reward] = Low_level_r(ghf, hf, candidate_fit, NFEs, Arm);  % 计算低层奖励(公式24)
    else
        reward = 0; % 重复解, 奖励为0
    end

end
