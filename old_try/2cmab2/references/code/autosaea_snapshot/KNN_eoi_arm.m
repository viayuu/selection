% =========================================================================
% KNN_eoi_arm: 组合臂 {KNN, L1-exploitation} 的执行函数
% 对应论文 IV.A节 第3)点 + 公式(11)(19)
%
% 工作流程:
%   1. 将当前种群按适应度分为L=5个等级, Level 1是最好的20%
%   2. 用KNN(K=1)分类器训练, 学习"什么样的解属于什么等级"
%   3. 用KNN预测所有DE子代的等级
%   4. 从预测为Level 1的子代中, 用L1-exploitation准则选候选解
%   5. 真实评估 + 计算奖励
%
% KNN模型 (论文公式(11)):
%   y_{KNN}(x) = mode({y(x_1), ..., y(x_K)})
%   K=1, 即新解的等级 = 离它最近的训练样本的等级
%
% L1-exploitation 准则 (论文公式(19)):
%   x* = argmin_{x∈X^1} max{ ||x-p||, p∈P^1 }
%   从预测为Level 1的子代中, 选离已知Level 1解集的"最远距离最小"的那个
%   直觉: 选一个"被已知好解包围的、在好区域中心"的点 → 深入开发(exploitation)
% =========================================================================
function  [hx, hf, reward ,NFEs, CE, gfs] = KNN_eoi_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, level, CE, gfs)

    %% --- 步骤1: 将种群分级 ---
    % 论文: "the current population is evenly divided into L levels"
    % level=5, 即分5级, 每级包含 N/5 = 20 个解
    sidx = 1:length(ghf);                   % ghx和ghf已按适应度排序(AutoSAEA.m中排过)
    train_label1 = ceil(sidx*level/length(ghf));  % 为每个解分配等级标签
    % 例: N=100, level=5
    %   第1-20个解(适应度最好) → label=1 (Level 1)
    %   第21-40个解 → label=2 (Level 2)
    %   ...
    %   第81-100个解(适应度最差) → label=5 (Level 5)

    %% --- 提取Level 1的父代解 (P^1) ---
    Parents_L1 = ghx(find(train_label1==1), :);  % P^1: 适应度最好的20%解集

    %% --- 步骤2: 训练KNN分类器 ---
    % 论文公式(11): K=1, 使用闵可夫斯基距离(p=2即欧氏距离)
    mdl = ClassificationKNN.fit(ghx, train_label1, 'Distance', 'minkowski');
    mdl.DistParameter = 2;                  % p=2, 即欧氏距离 ||x-y||_2

    %% --- 步骤3: 用KNN预测所有子代的等级 ---
    label = predict(mdl, offspring);          % 预测每个子代属于哪个等级(1-5)

    %% --- 步骤4: L1-exploitation 选择 ---
    % 从预测为Level 1的子代中选择
    select_pp = find(label == min(label));    % 找出预测标签为1(最好等级)的子代索引
                                             % 这些子代构成 X^1 (论文公式(19)中的X^1)

    % 计算每个L1候选到所有L1父代的距离
    for ii3 = 1: length(select_pp)           % 遍历每个预测为L1的子代
        for j = 1 : size(Parents_L1, 1)      % 遍历每个L1父代
            % 计算欧氏距离 ||offspring - parent||_2
            dist(ii3,j) = sqrt(sum((offspring(select_pp(ii3), :)-Parents_L1(j, :)).^2, 2));
        end
    end

    % L1-exploitation 准则, 对应论文公式(19):
    % x* = argmin_{x∈X^1} max{ ||x-p||, p∈P^1 }
    max_dist = max(dist,[],2);               % 每个L1候选到最远L1父代的距离
                                             % max{||x-p||, p∈P^1}
    [~, in] = min(max_dist);                 % 选这个最大距离最小的候选
                                             % argmin max{...}
    % 直觉: 选的是"被已知好解包围得最紧的"那个候选, 即好区域的中心 → 深入开发
    candidate_position = offspring(select_pp(in),:);  % 候选解 x_t

    %% --- 去重检查 + 真实评估 + 奖励计算 ---
    [~,ih,~] = intersect(hx,candidate_position,'rows');
    if isempty(ih)==1
       candidate_fit=FUN(candidate_position);   % 真实评估
       NFEs = NFEs + 1;

       hx=[hx; candidate_position];  hf=[hf, candidate_fit];  % 更新数据库

       CE(NFEs,:)=[NFEs,candidate_fit];
       gfs(1,NFEs)=min(CE(1:NFEs,2));

       Arm = num2str('KNN_L1-exploitation ');
       [reward] = Low_level_r(ghf, hf, candidate_fit, NFEs, Arm);  % 低层奖励(公式24)
    else
       reward = 0;
    end

end
