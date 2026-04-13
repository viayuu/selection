% =========================================================================
% KNN_eor_arm: 组合臂 {KNN, L1-exploration} 的执行函数
% 对应论文 IV.A节 第3)点 + 公式(11)(20)
%
% 工作流程与 KNN_eoi_arm 完全相同, 唯一区别在步骤4的选择准则:
%   - L1-exploitation(公式19): argmin max{||x-p||} → 选离好解最近的 → 深入开发
%   - L1-exploration(公式20):  argmax min{||x-p||} → 选离好解最远的 → 探索新区域
%
% L1-exploration 准则 (论文公式(20)):
%   x* = argmax_{x∈X^1} min{ ||x-p||, p∈P^1 }
%   从预测为Level 1的子代中, 选离已知Level 1解集的"最近距离最大"的那个
%   直觉: 选一个"虽然KNN预测它是好的, 但远离已知好解"的点 → 探索(exploration)
% =========================================================================
function  [hx, hf, reward ,NFEs, CE, gfs] = Knn_eor_arm(ghx,ghf, offspring, hx, hf, FUN, NFEs, level, CE, gfs)

    %% --- 步骤1: 将种群分级 (同 KNN_eoi_arm) ---
    sidx = 1:length(ghf);
    train_label1 = ceil(sidx*level/length(ghf));  % 分5级标签
    Parents_L1 = ghx(find(train_label1==1), :);   % P^1: Level 1父代

    %% --- 步骤2: 训练KNN分类器 (同 KNN_eoi_arm) ---
    mdl = ClassificationKNN.fit(ghx, train_label1, 'Distance', 'minkowski');
    mdl.DistParameter = 2;                  % 欧氏距离

    %% --- 步骤3: 预测子代等级 ---
    label = predict(mdl, offspring);
    select_pp = find(label == min(label));    % 预测为Level 1的子代索引 (构成 X^1)

    %% --- 步骤4: L1-exploration 选择 (与 exploitation 的关键区别!) ---
    % 计算每个L1候选到所有L1父代的距离
    for ii3 = 1: length(select_pp)
        for j = 1 : size(Parents_L1, 1)
            dist(ii3,j) = sqrt(sum((offspring(select_pp(ii3), :)-Parents_L1(j, :)).^2, 2));
        end
    end

    % L1-exploration 准则, 对应论文公式(20):
    % x* = argmax_{x∈X^1} min{ ||x-p||, p∈P^1 }
    max_dist = min(dist,[],2);               % 每个L1候选到最近L1父代的距离
                                             % 注意这里是 min (与exploitation中的max相反!)
                                             % min{||x-p||, p∈P^1}
    [~, in] = max(max_dist);                 % 选这个最小距离最大的候选
                                             % argmax min{...}
    % 直觉: 选的是"离所有已知好解都尽量远的"那个候选
    % 虽然KNN预测它也是好的(Level 1), 但它在一个尚未被充分探索的区域 → 探索
    candidate_position = offspring(select_pp(in),:);  % 候选解 x_t

    %% --- 去重检查 + 真实评估 + 奖励计算 ---
    [~,ih,~] = intersect(hx,candidate_position,'rows');
    if isempty(ih)==1
       candidate_fit=FUN(candidate_position);   % 真实评估
       NFEs = NFEs + 1;

       hx=[hx; candidate_position];  hf=[hf, candidate_fit];

       CE(NFEs,:)=[NFEs,candidate_fit];
       if mod (NFEs,1)==0                       % 每次都记录(mod(NFEs,1)==0 恒为真)
          cs1=NFEs/1; gfs(1,cs1)=min(CE(1:NFEs,2));
       end

       Arm = num2str('KNN_L1-exploration ');
       [reward] = Low_level_r(ghf, hf, candidate_fit, NFEs, Arm);
    else
       reward = 0;
    end

end
