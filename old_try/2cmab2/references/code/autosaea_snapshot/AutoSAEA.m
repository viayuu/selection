% =========================================================================
% AutoSAEA 主函数 —— 对应论文 Algorithm 1
% 论文标题: Surrogate-Assisted Evolutionary Algorithm With Model and Infill
%          Criterion Auto-Configuration (IEEE TEVC, 2024)
%
% 核心思想: 将"选代理模型"和"选填充准则"建模为两层多臂老虎机问题(TL-MAB)
%   高层摇臂(High-level arms) = 代理模型: {GP, RBF, PRS, KNN}
%   低层摇臂(Low-level arms) = 填充准则: {LCB, EI, prescreening, local search, ...}
%   通过 TL-UCB 策略在线选择, 通过 TL-R 奖励机制更新各摇臂价值
%
% 输入:
%   FUN - 目标函数句柄(昂贵的真实评估函数, 对应论文公式(1)中的 f(x))
%   D   - 决策变量维度(对应论文公式(1)中的 D)
%   LB  - 决策变量下界向量(对应论文公式(1)中的 x_l)
%   UB  - 决策变量上界向量(对应论文公式(1)中的 x_u)
%
% 输出:
%   hf     - 所有已评估解的真实适应度值(历史数据库 D 中的 f(x) 值)
%   MaxFEs - 最大函数评估次数
%   gfs    - 每次评估后的全局最优适应度值(用于绘制收敛曲线)
%   s_l_s  - (未使用)
% =========================================================================
function [hf,MaxFEs,gfs,s_l_s] = AutoSAEA(FUN,D,LB,UB)

%% ================== 初始化阶段 (对应 Algorithm 1, 第7-13行) ==================
% --- DE算子参数设置 (对应论文公式(14)(15)中的参数) ---
F=0.5;                      % DE缩放因子(论文: F is set to 0.5)
CR=0.9;                     % DE交叉率(论文: CR is set to 0.9)

% --- TL-MAB参数设置 ---
apha = 2.5;                 % UCB探索参数α (对应论文公式(22)(23)中的α, 控制探索-利用平衡)
                            % α越大, 越倾向于探索较少被选择的摇臂
level = 5;                  % KNN分类器的分级数L (论文: the number of levels is set to 5)
                            % 种群被等分为5级, 最好的20%为Level 1
initial_sample_size=100;    % 种群大小N (对应论文Algorithm 1第2行的N)
NFEs=0;                     % 当前已消耗的函数评估次数(Function Evaluations)
MaxFEs=1000;                % 最大函数评估次数(对应论文Algorithm 1第3行的MaxFEs)
CE=zeros(MaxFEs,2);         % 记录每次评估的[评估次数, 适应度值], 用于绘制收敛曲线
gfs = zeros(1,MaxFEs);      % 记录每次评估后的全局最优值
VRmin=repmat(LB,initial_sample_size,1);  % 将下界复制N行, 用于DE算子的边界约束(N×D矩阵)
VRmax=repmat(UB,initial_sample_size,1);  % 将上界复制N行, 用于DE算子的边界约束(N×D矩阵)

%% ================== 生成初始样本 (对应 Algorithm 1, 第8-9行) ==================
% 论文: "Use LHS to sample N solutions in the search space"
% LHS(拉丁超立方采样)是一种空间填充采样方法, 比纯随机采样能更均匀地覆盖搜索空间
gs = initial_sample_size;   % gs = N = 100, 种群大小
sam=repmat(LB,initial_sample_size,1) + ...                          % 下界
    (repmat(UB,initial_sample_size,1)-repmat(LB,initial_sample_size,1)) ...  % (上界-下界)
    .* lhsdesign(initial_sample_size,D);                             % × LHS采样矩阵[0,1]^(N×D)
% 结果: sam 是 N×D 的矩阵, 每行是一个D维的初始解, 值在[LB, UB]范围内

% --- 对初始样本进行真实的昂贵评估 (对应 Algorithm 1, 第9行 "Evaluate") ---
for i=1:initial_sample_size
    fitness(i) = FUN(sam(i,:));         % 调用真实目标函数评估第i个初始解
    NFEs=NFEs+1;                         % 函数评估次数+1
    CE(NFEs,:)=[NFEs,fitness(i)];        % 记录本次评估结果
    gfs(1,NFEs)=min(CE(1:NFEs,2));       % 更新截至目前的全局最优适应度值
end
% 初始化评估完成后, NFEs = N = 100

hx=sam;          % hx: 历史解数据库(对应论文的数据库 D 中的决策向量), 初始为N个LHS样本
hf=fitness;      % hf: 历史适应度数据库(对应论文的数据库 D 中的 f(x) 值)
id = 0;          % 迭代计数器(对应论文Algorithm 1中的 t)

%% ================== 奖励相关变量初始化 (对应 Algorithm 1, 第10-11行) ==================
% 论文: "Set Q_a^H(1)=0 for all a∈A^H, and Q_a^L=0 for all a∈A^L"
% 论文: "Set T_a^H(1)=0 for all a∈A^H, and T_a^L=0 for all a∈A^L"

% --- 高层摇臂(模型)的奖励记录 ---
% 每个模型关联的所有低层摇臂的奖励历史(用于计算高层Q值 = mean(所有关联低层奖励))
% 对应论文公式(30): Q_{a^H}(t) = Σ Q_{a^L}(t) / |A_{a^H}^L|
rbf_model = [];     % RBF模型的所有关联低层奖励 (关联: prescreening + local search)
gp_model =[];       % GP模型的所有关联低层奖励 (关联: LCB + EI)
knn_model =[];      % KNN模型的所有关联低层奖励 (关联: L1-exploitation + L1-exploration)
prs_model =[];      % PRS模型的所有关联低层奖励 (关联: prescreening + local search)

% --- 低层摇臂(填充准则)的奖励历史 ---
% 每个低层摇臂的所有历史奖励(用于计算低层Q值 = mean(历史奖励), 对应论文公式(25))
Save_rp =[];        % a_3^L: RBF + prescreening 的历史奖励
Save_rl = [];       % a_4^L: RBF + local search 的历史奖励
Save_gl =[];        % a_1^L: GP + LCB 的历史奖励
Save_ge = [];       % a_2^L: GP + EI 的历史奖励
Save_pp =[];        % a_5^L: PRS + prescreening 的历史奖励
Save_pl =[];        % a_6^L: PRS + local search 的历史奖励
Save_ki =[];        % a_7^L: KNN + L1-exploitation 的历史奖励
Save_ko =[];        % a_8^L: KNN + L1-exploration 的历史奖励

%% ================== 组合臂集合初始化 (对应 Algorithm 1, 第12行) ==================
% 论文: "Set CA = {(GP,LCB), (GP,EI), (RBF,prescreening), (RBF,local search),
%        (PRS,prescreening), (PRS,local search), (KNN,L1-exploitation), (KNN,L1-exploration)}"
% 总共8种合法的(模型, 填充准则)组合臂
num_arm = 8;            % 组合臂总数 |CA| = 8
in1 = randperm(num_arm); % 随机排列(代码中未实际使用此排列, 热身阶段按固定顺序执行)

%% ================== 主优化循环 (对应 Algorithm 1, 第14-29行) ==================
% 论文: "while FEs < MaxFEs do"
% 每次迭代消耗1次真实函数评估(FE), 从第101次到第1000次
while NFEs < MaxFEs
      id = id +1;  % 迭代计数器+1 (对应论文的 t = t+1, Algorithm 1 第28行)

      %% --- 选择当前种群P (对应 Algorithm 1, 第15行) ---
      % 论文: "Select N best solutions from D as the population P"
      % 从历史数据库中选出适应度最好的N个解作为当前种群
      [~, sort_index] = sort(hf);                  % 将所有历史适应度值升序排列(越小越好)
      ghf=hf(sort_index(1:gs));                     % 取前N个最优适应度值作为当前种群适应度
      ghx=hx(sort_index(1:gs),:);                   % 取对应的前N个最优解作为当前种群P
      % 注: ghx 已按适应度排序, ghx(1,:)是当前最优解 x_best

      %% ================== 热身阶段 (对应 Algorithm 1, 第16-18行) ==================
      % 论文: "if t ≤ |CA| then select the t-th combinatorial arm in CA"
      % 前8轮依次尝试每种组合臂各一次, 确保每个摇臂至少被选择一次
      % 这是UCB策略的必要条件: T_a(t) ≠ 0, 否则UCB公式中会出现除以零
      if id <= num_arm
         if id == 1
            %% --- 第1轮: 组合臂 {RBF, prescreening} ---
            % 高层臂 a^H = RBF, 低层臂 a^L = prescreening (对应论文公式(17))
            % 先用DE生成N个子代, 再用RBF模型预测, 选预测最优的子代做真实评估
            offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin); % 用DE算子生成N个子代(公式14-16)
            [hx, hf, reward_rp, NFEs, CE, gfs] = RBF_pre_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs);
            Save_rp = [Save_rp, reward_rp];     % 记录该低层臂(RBF prescreening)的奖励
            rbf_model = [rbf_model;reward_rp];   % 记录到RBF高层臂的关联奖励中

         elseif id ==2
            %% --- 第2轮: 组合臂 {GP, LCB} ---
            % 高层臂 a^H = GP, 低层臂 a^L = LCB (对应论文公式(12))
            % 用GP模型预测每个子代的均值和方差, 选LCB值最小的做真实评估
            offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin); % 用DE算子生成N个子代
            [hx, hf, reward_gl, NFEs, CE, gfs] = GP_lcb_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs);
            Save_gl = [Save_gl, reward_gl];      % 记录该低层臂(GP LCB)的奖励
            gp_model =[gp_model; reward_gl];      % 记录到GP高层臂的关联奖励中

         elseif id == 3
            %% --- 第3轮: 组合臂 {RBF, local search} ---
            % 高层臂 a^H = RBF, 低层臂 a^L = local search (对应论文公式(17)+(18))
            % 在当前种群的包围盒内, 用DE优化RBF代理模型, 找到模型最优解做真实评估
            % 注意: local search 不需要外部生成子代, 内部自行用DE搜索
            [hx, hf, reward_rl, NFEs,  CE, gfs] = RBF_ls_arm(ghx, ghf, hx, hf, FUN, NFEs, CE, gfs);
            Save_rl = [Save_rl, reward_rl];      % 记录该低层臂(RBF local search)的奖励
            rbf_model = [rbf_model;reward_rl];    % 记录到RBF高层臂的关联奖励中

         elseif id == 4
            %% --- 第4轮: 组合臂 {GP, EI} ---
            % 高层臂 a^H = GP, 低层臂 a^L = EI (对应论文公式(13))
            % 用GP模型计算每个子代的期望改进(Expected Improvement), 选EI最大的做真实评估
            offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin); % 用DE算子生成N个子代
            [hx, hf, reward_ge, NFEs, CE, gfs] = GP_ei_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs);
            Save_ge = [Save_ge, reward_ge];      % 记录该低层臂(GP EI)的奖励
            gp_model =[gp_model; reward_ge];      % 记录到GP高层臂的关联奖励中

         elseif id == 5
            %% --- 第5轮: 组合臂 {PRS, prescreening} ---
            % 高层臂 a^H = PRS, 低层臂 a^L = prescreening (对应论文公式(17), 但用PRS预测)
            offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin); % 用DE算子生成N个子代
            [hx, hf, reward_pp, NFEs, CE, gfs] = PRS_pre_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs);
            Save_pp = [Save_pp, reward_pp];      % 记录该低层臂(PRS prescreening)的奖励
            prs_model = [prs_model;reward_pp];    % 记录到PRS高层臂的关联奖励中

         elseif id == 6
            %% --- 第6轮: 组合臂 {PRS, local search} ---
            % 高层臂 a^H = PRS, 低层臂 a^L = local search
            [hx, hf, reward_pl, NFEs, CE, gfs] = PRS_ls_arm(ghx, ghf, hx, hf, FUN, NFEs, CE, gfs);
            Save_pl = [Save_pl, reward_pl];      % 记录该低层臂(PRS local search)的奖励
            prs_model = [prs_model;reward_pl];    % 记录到PRS高层臂的关联奖励中

         elseif id == 7
            %% --- 第7轮: 组合臂 {KNN, L1-exploitation} ---
            % 高层臂 a^H = KNN, 低层臂 a^L = L1-exploitation (对应论文公式(19))
            offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin); % 用DE算子生成N个子代
            [hx, hf, reward_ki ,NFEs, CE, gfs] = KNN_eoi_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, level, CE, gfs);
            Save_ki = [Save_ki, reward_ki];      % 记录该低层臂(KNN L1-exploitation)的奖励
            knn_model = [knn_model;reward_ki];    % 记录到KNN高层臂的关联奖励中

         elseif id == 8
            %% --- 第8轮: 组合臂 {KNN, L1-exploration} ---
            % 高层臂 a^H = KNN, 低层臂 a^L = L1-exploration (对应论文公式(20))
            offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin); % 用DE算子生成N个子代
            [hx, hf, reward_ko ,NFEs, CE, gfs] = KNN_eor_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, level, CE, gfs);
            Save_ko = [Save_ko, reward_ko];      % 记录该低层臂(KNN L1-exploration)的奖励
            knn_model = [knn_model;reward_ko];    % 记录到KNN高层臂的关联奖励中

         end

      else
      %% ================== TL-UCB选择阶段 (对应 Algorithm 1 第20行, Algorithm 2) ==================
      % 论文: "Choose a_t^H and a_t^L by TL-UCB"
      % 热身阶段结束后(id > 8), 每轮通过TL-UCB策略选择高层臂和低层臂

          %% --- 第一步: 选择高层臂(代理模型) ---
          % 对应论文公式(22): a_t^H = argmax_{a∈A^H} [ Q_a^H(t) + sqrt(α·ln(t)/T_a^H(t)) ]
          % 遍历4个高层臂{RBF, GP, PRS, KNN}, 计算各自的UCB值
          for i = 1 : 4
              if i ==1
                % --- RBF模型的UCB计算 ---
                sum_reward = rbf_model;           % RBF的所有关联低层奖励历史
                q_value_m = mean(rbf_model);      % Q_RBF^H(t) = 所有关联低层奖励的均值
                                                  % 对应论文公式(30): Q_{a^H} = Σ Q_{a^L} / |A_{a^H}^L|
              elseif i ==2
                % --- GP模型的UCB计算 ---
                sum_reward = gp_model;
                q_value_m = mean(gp_model);       % Q_GP^H(t)
              elseif i ==3
                % --- PRS模型的UCB计算 ---
                sum_reward = prs_model;
                q_value_m = mean(prs_model);      % Q_PRS^H(t)
              elseif i ==4
                % --- KNN模型的UCB计算 ---
                sum_reward = knn_model;
                q_value_m = mean(knn_model);      % Q_KNN^H(t)
              end
              % 调用TL_UCB函数计算: UCB值 = Q值 + 探索项
              % sum_reward的长度 = T_a^H(t), 即该高层臂被选择的总次数
              U_model_value(i) = TL_UCB(sum_reward, id, q_value_m, apha);
          end

          % 选择UCB值最大的高层臂
          idx = find(U_model_value == max(U_model_value));
          if length(idx) >= 1   % 如果有多个UCB值并列最大, 随机选一个(打破平局)
             in6 = randperm(size(idx, 1));
             idx = idx(in6(1));
          end

          %% --- 第二步: 根据选中的高层臂, 选择低层臂(填充准则) ---
          % 对应论文公式(23): a_t^L = argmax_{a∈A_{a_t^H}^L} [ Q_a^L(t) + sqrt(α·ln(t)/T_a^L(t)) ]
          % 只在被选高层臂关联的低层臂中选择(对应论文Fig.2的关联关系)

          if idx == 1
             %% === 高层臂 RBF 被选中 ===
             % 关联的低层臂: A_{RBF}^L = {prescreening(a_3^L), local search(a_4^L)}
             for i =1 : 2
                 if i == 1
                    % RBF + prescreening 的UCB计算
                    sum_reward = Save_rp;             % prescreening的所有历史奖励
                    q_value = mean(Save_rp);          % Q_{prescreening}^L(t) = 历史奖励均值

                 elseif i ==2
                    % RBF + local search 的UCB计算
                    sum_reward = Save_rl;
                    q_value = mean(Save_rl);          % Q_{local_search}^L(t)

                 end
                 U_rbf_value(i) = TL_UCB(sum_reward, id, q_value, apha);  % 计算低层臂的UCB值
             end
             % 选择UCB值最大的低层臂
             idx2 = find(U_rbf_value == max(U_rbf_value));
             if length(idx2) >= 1   % 平局随机打破
                in6 = randperm(size(idx2, 1));
                idx2 = idx2(in6(1));
             end
             if idx2 == 1
                  %% --- 执行组合臂 {RBF, prescreening} ---
                  % 对应论文IV.A节第2)点: "if a_t^L = prescreening, 选预测最优的子代"
                offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin);  % DE生成N个子代
                [hx, hf, reward_rp, NFEs,  CE, gfs] = RBF_pre_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs);
                Save_rp = [Save_rp, reward_rp];       % 更新低层臂奖励历史 (对应Algorithm 1第27行)
                rbf_model = [rbf_model;reward_rp];     % 更新高层臂关联奖励

              elseif idx2 == 2
                  %% --- 执行组合臂 {RBF, local search} ---
                  % 对应论文IV.A节第2)点: "if a_t^L = local search, 用DE优化RBF模型"
                 [hx, hf, reward_rl,  NFEs,  CE, gfs] = RBF_ls_arm(ghx, ghf, hx, hf, FUN, NFEs, CE, gfs);
                 Save_rl = [Save_rl, reward_rl];
                 rbf_model = [rbf_model; reward_rl];

              end
          elseif idx == 2
             %% === 高层臂 GP 被选中 ===
             % 关联的低层臂: A_{GP}^L = {LCB(a_1^L), EI(a_2^L)}
             for i = 1 : 2
                 if i == 1
                    % GP + LCB 的UCB计算
                    sum_reward = Save_gl;
                    q_value =  mean(Save_gl);         % Q_{LCB}^L(t)

                 elseif i ==2
                    % GP + EI 的UCB计算
                    sum_reward = Save_ge;
                    q_value = mean(Save_ge);           % Q_{EI}^L(t)

                 end
                 U_gp_value(i) = TL_UCB(sum_reward, id, q_value, apha);  % 计算低层臂的UCB值
             end
             idx3 = find(U_gp_value == max(U_gp_value));
             if length(idx3) >= 1
                 in6 = randperm(size(idx3, 1));
                 idx3 = idx3(in6(1));
             end
             if idx3 == 1
                  %% --- 执行组合臂 {GP, LCB} ---
                  % 对应论文IV.A节第1)点: "选LCB值最优的子代"
                  offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin);
                  [hx, hf, reward_gl, NFEs, CE, gfs] = GP_lcb_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs);
                  Save_gl = [Save_gl, reward_gl];
                  gp_model =[gp_model; reward_gl];

             elseif idx3 == 2
                 %% --- 执行组合臂 {GP, EI} ---
                 % 对应论文IV.A节第1)点: "选EI值最优的子代"
                 offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin);
                 [hx, hf, reward_ge, NFEs, CE, gfs] = GP_ei_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs);
                 Save_ge = [Save_ge, reward_ge];
                 gp_model =[gp_model; reward_ge];

             end
          elseif idx == 3
             %% === 高层臂 PRS 被选中 ===
             % 关联的低层臂: A_{PRS}^L = {prescreening(a_5^L), local search(a_6^L)}
              for i = 1 : 2
                  if i == 1
                    % PRS + prescreening 的UCB计算
                    sum_reward = Save_pp;
                    q_value = mean(Save_pp);           % Q_{PRS_prescreening}^L(t)
                  elseif i ==2
                    % PRS + local search 的UCB计算
                    sum_reward = Save_pl;
                    q_value = mean(Save_pl);            % Q_{PRS_local_search}^L(t)
                  end
                  U_prs_value(i) = TL_UCB(sum_reward, id, q_value, apha);
              end
              idx4 = find(U_prs_value == max(U_prs_value));
              if length(idx4) >= 1
                 in6 = randperm(size(idx4, 1));
                 idx4 = idx4(in6(1));
              end
              if idx4 == 1
                  %% --- 执行组合臂 {PRS, Prescreening} ---
                 offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin);
                 [hx, hf, reward_pp, NFEs, CE, gfs] = PRS_pre_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs);
                 Save_pp = [Save_pp, reward_pp];
                 prs_model = [prs_model;reward_pp];

              elseif idx4 == 2
                 %% --- 执行组合臂 {PRS, local search} ---
                [hx, hf, reward_pl, NFEs, CE, gfs] = PRS_ls_arm(ghx, ghf, hx, hf, FUN, NFEs, CE, gfs);
                Save_pl = [Save_pl, reward_pl];
                prs_model = [prs_model;reward_pl];

              end
          elseif idx == 4
             %% === 高层臂 KNN 被选中 ===
             % 关联的低层臂: A_{KNN}^L = {L1-exploitation(a_7^L), L1-exploration(a_8^L)}
              for i = 1 : 2
                  if i == 1
                    % KNN + L1-exploitation 的UCB计算
                    sum_reward = Save_ki ;
                    q_value = mean(Save_ki);            % Q_{L1-exploitation}^L(t)
                 elseif i ==2
                    % KNN + L1-exploration 的UCB计算
                    sum_reward = Save_ko ;
                    q_value = mean(Save_ko);             % Q_{L1-exploration}^L(t)
                 end
                 U_knn_value(i) = TL_UCB(sum_reward, id, q_value, apha);
              end
              idx5 = find(U_knn_value == max(U_knn_value));
              if length(idx5) >= 1
                  in6 = randperm(size(idx5, 1));
                  idx5 = idx5(in6(1));
              end

              if idx5 == 1
                   %% --- 执行组合臂 {KNN, L1-exploitation} ---
                   % 对应论文公式(19): 从预测为Level 1的子代中选离已知好解最近的
                 offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin);
                 [hx, hf, reward_ki ,NFEs, CE, gfs] = KNN_eoi_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, level, CE, gfs);
                 Save_ki = [Save_ki, reward_ki];
                 knn_model = [knn_model;reward_ki];

              elseif idx5 == 2
                  %% --- 执行组合臂 {KNN, L1-exploration} ---
                  % 对应论文公式(20): 从预测为Level 1的子代中选离已知好解最远的
                 offspring = DEoperator(ghx,gs,D,ghx,F,CR,VRmax,VRmin);
                 [hx, hf, reward_ko, NFEs, CE, gfs] = KNN_eor_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, level, CE, gfs);
                 Save_ko = [Save_ko , reward_ko];
                 knn_model = [knn_model;reward_ko];

              end
              if NFEs > 1000   % 安全检查: 如果超过最大评估次数则退出
                 break
              end
          end
     end
end
% 论文 Algorithm 1 第30行: "Output: The best solution in D"
% 返回 hf (包含所有历史适应度值), 调用者可用 min(hf) 获取最优解
end
