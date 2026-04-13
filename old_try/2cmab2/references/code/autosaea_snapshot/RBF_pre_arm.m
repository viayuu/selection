% =========================================================================
% RBF_pre_arm: 组合臂 {RBF, prescreening} 的执行函数
% 对应论文 IV.A节 第2)点 + 公式(7)(8)(17)
%
% 工作流程(对应论文的描述):
%   1. 用当前种群训练RBF代理模型 (论文公式(7)(8))
%   2. 用RBF模型预测所有DE子代的适应度
%   3. 选预测适应度最好(最小)的子代作为候选解 (论文公式(17), prescreening)
%   4. 对候选解进行真实的昂贵评估
%   5. 计算低层奖励 (论文公式(24))
%
% 论文公式(17) [prescreening]:
%   x* = argmin_{x∈X} f̂_{RBF}(x)
%   其中 X 是DE生成的N个子代的有限集合
%   "如果X是有限解集, 公式(17)属于prescreening"
%
% 输入:
%   ghx       - 当前种群的决策变量矩阵 (N×D)
%   ghf       - 当前种群的适应度值向量 (1×N)
%   offspring - DE算子生成的子代矩阵 (N×D)
%   hx        - 历史解数据库 (所有已评估解的决策变量)
%   hf        - 历史适应度数据库
%   FUN       - 真实目标函数句柄
%   NFEs      - 当前函数评估次数
%   CE, gfs   - 收敛记录变量
%
% 输出:
%   hx, hf    - 更新后的历史数据库 (加入新评估的解)
%   reward    - 低层奖励 r_{a_t^L} (论文公式(24))
%   NFEs      - 更新后的函数评估次数
%   CE, gfs   - 更新后的收敛记录
% =========================================================================
function [hx, hf, reward,  NFEs,  CE, gfs] = RBF_pre_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs)

    %% --- 步骤1: 训练RBF代理模型 ---
    % 对应论文公式(7): f̂_{RBF}(x) = Σ wᵢ·φ(||x-xᵢ||) + b₀ + Σ bⱼ·xⱼ
    % 使用立方基函数(CUB): φ(r) = r³, 对应论文中的 φ(||x-xᵢ||) = (||x-xᵢ||)³
    % ghx: 训练输入(当前种群的决策变量), ghf': 训练输出(适应度值, 转为列向量)
    srgtOPT=srgtsRBFSetOptions(ghx, ghf', @my_rbfbuild, [],'CUB', 0.0002,1);
    srgtSRGT = srgtsRBFFit(srgtOPT);       % 求解RBF模型参数w和b (论文公式(8))

    %% --- 步骤2: 用RBF模型预测所有子代的适应度 ---
    % 计算 f̂_{RBF}(offspring_i) for each offspring_i
    fitnessModel= my_rbfpredict(srgtSRGT.RBF_Model, srgtSRGT.P, offspring);

    %% --- 步骤3: Prescreening —— 选预测最优的子代 ---
    % 对应论文公式(17): x* = argmin_{x∈X} f̂_{RBF}(x)
    [~,sidx]=min(fitnessModel);             % 找到预测适应度最小(最好)的子代索引
    candidate_position = offspring(sidx, :); % 该子代即为候选解 x_t

    %% --- 步骤3.5: 去重检查 ---
    % 检查候选解是否已经在历史数据库中(避免重复评估浪费预算)
    [~,ih,~] = intersect(hx,candidate_position,'rows');

    if isempty(ih)==1       % 如果候选解是新的(不在数据库中)
        %% --- 步骤4: 真实的昂贵评估 ---
        % 对应论文 Algorithm 1 第23行: "Evaluate x_t"
        candidate_fit=FUN(candidate_position);  % 调用真实目标函数 f(x_t)
        NFEs = NFEs + 1;                         % 消耗1次真实函数评估

        %% --- 步骤4.5: 更新历史数据库 ---
        % 对应论文 Algorithm 1 第24行: "D = D ∪ x_t"
        hx=[hx; candidate_position];             % 将新解的决策变量加入数据库
        hf=[hf, candidate_fit];                   % 将新解的适应度值加入数据库

        % 更新收敛记录
        CE(NFEs,:)=[NFEs,candidate_fit];
        gfs(1,NFEs)=min(CE(1:NFEs,2));            % 记录截至当前的全局最优值

        %% --- 步骤5: 计算低层奖励 ---
        % 对应论文 Algorithm 3 第7行, 公式(24)
        Arm = num2str('RBF_prescreening ');
        [reward] = Low_level_r(ghf, hf, candidate_fit, NFEs, Arm);
    else
        reward = 0; % 候选解已存在于数据库中, 无新信息, 奖励为0
    end

end
