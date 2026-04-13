% =========================================================================
% DE_optimizer: 差分进化优化器 —— 用于 local search 填充准则
% 对应论文公式(17)+(18), 即在代理模型上进行局部搜索
%
% 作用: 在当前种群的包围盒[lb, ub]内, 用DE算法优化代理模型(RBF或PRS)
%       找到代理模型预测的最优解, 作为候选解进行真实评估
%       注意: 这里所有的"函数评估"都是在代理模型上进行的(计算成本很低)
%       不消耗真实的昂贵函数评估次数!
%
% 论文: "DE local search is conducted with a population size of N
%        and a maximum number of 100D + 1000 generations" [25]
%
% 论文公式(18): 搜索空间 X = [lb, ub]
%   lb_j = min{x_{i,j}, x_i ∈ P}, ub_j = max{x_{i,j}, x_i ∈ P}
%   即搜索范围限制在当前种群的变量包围盒内(局部搜索)
%
% 输入:
%   Dim       - 决策变量维度 D
%   Max_NFEs  - 最大代理模型评估次数 (= 100·D + 1000)
%   srgtSRGT  - 训练好的代理模型结构体 (RBF 或 PRS)
%   minerror  - 最小改进阈值 (= 1e-20), 连续10代改进小于此值则提前停止
%   ghx       - 当前种群矩阵 (N×D), 用于确定搜索范围
%   flag      - 模型类型标志: 1=RBF模型, 0=PRS模型
%
% 输出:
%   bestP          - 代理模型上找到的最优解(D维向量)
%   bestFitness    - 该最优解在代理模型上的预测适应度
%   bestP_position - 最优解在种群中的位置索引
%
% 来源注释: 代码借鉴自 Zhen et al. "Two-stage Data-driven Evolutionary
%          Optimization for High-dimensional Expensive Problems", IEEE TCYB, 2021
% =========================================================================
function [bestP,bestFitness, bestP_position] = DE_optimizer(Dim, Max_NFEs,srgtSRGT,minerror,ghx, flag)
    time_begin=tic;                         % 计时开始
    n = Dim;                                % 维度
    NP = size(ghx, 1);                      % 种群大小N (与主算法的种群大小相同)
    flag_er=0;                              % 连续未改进计数器(用于提前停止)

    %% --- 确定搜索范围 (对应论文公式(18)) ---
    % 搜索范围 = 当前种群在每个维度上的[最小值, 最大值]
    % 这实现了"局部搜索"的概念: 只在当前已知好解的附近区域搜索
    lu = [min(ghx); max(ghx)];              % 2×D矩阵: 第1行=各维最小值, 第2行=各维最大值
    LowerBound = lu(1, :);                  % lb = (l_1, ..., l_D), 论文公式(18)
    UpperBound = lu(2, :);                  % ub = (u_1, ..., u_D), 论文公式(18)

    %% --- DE参数设置 ---
    G=1;                                    % 当前代数计数器
    F=0.5;                                  % DE缩放因子
    CR=0.9;                                 % DE交叉率

    UB=UpperBound;
    LB=LowerBound;
    UB=repmat((UB),NP,1);                   % 将上界复制NP行 (NP×D矩阵, 方便矩阵运算)
    LB=repmat((LB),NP,1);                   % 将下界复制NP行

    %% --- 在搜索范围内随机初始化DE种群 ---
    P=(UB-LB).*rand(NP,Dim)+LB;            % 随机生成NP个D维的初始解

    %% --- 用代理模型评估初始种群(不消耗真实FEs!) ---
    if flag ==1
       fitnessP= my_rbfpredict(srgtSRGT.RBF_Model, srgtSRGT.P, P); % 用RBF模型预测适应度
    else
       fitnessP  = srgtsPRSEvaluate(P, srgtSRGT);                   % 用PRS模型预测适应度
    end
    NFEs=NP;                                % 代理模型评估次数(非真实FEs)
    [fitnessBestP,indexBestP]=min(fitnessP); % 找到初始种群中代理模型预测最优的解
    bestP_position = indexBestP;
    bestP=P(indexBestP,:);                  % 当前最优解
    recRMSE(1:NP)=fitnessP;

    %% --- DE主循环: 在代理模型上优化 ---
    % 最多运行 100D+1000 代(每代NP次代理模型评估)
    while NFEs<Max_NFEs

        fitnessBestP_old = fitnessBestP;    % 记录上一代的最优值(用于判断是否改进)

        % --- 对每个个体进行变异和交叉 ---
        for i=1:NP
            % 随机选3个互不相同且不等于i的个体(与DEoperator.m类似)
            k0=randi([1,NP]);
            while(k0==i)
                k0=randi([1,NP]);
            end
            P1=P(k0,:);
            k1=randi([1,NP]);
            while(k1==i||k1==k0)
                k1=randi([1,NP]);
            end
            P2=P(k1,:);
            k2=randi([1,NP]);
            while(k2==i||k2==k1||k2==k0)
                k2=randi([1,NP]);
            end
            P3=P(k2,:);

            % --- DE/best/1 变异策略 ---
            % V = bestP + F·(P1 - P2)
            % 注意: 这里用的是 DE/best/1, 与主算法DEoperator中的 DE/current-to-best/1 不同
            % DE/best/1 更激进, 总是以当前最优解为基向量, 适合在代理模型上快速收敛
            V(i,:)= bestP+F.*(P1-P2);

            % 边界修复(同论文公式(16))
            for j=1:Dim
              if (V(i,j)>UB(i,j)||V(i,j)<LB(i,j))
                 V(i,j)=LB(i,j)+rand*(UB(i,j)-LB(i,j));
              end
            end

            % 二项交叉(同论文公式(15))
            jrand=randi([1,Dim]);
            for j=1:Dim
                k3=rand;
                if(k3<=CR||j==jrand)
                    U(i,j)=V(i,j);     % 从变异向量继承
                else
                    U(i,j)=P(i,j);     % 从父代保留
                end
            end
        end

        %% --- 用代理模型评估子代(不消耗真实FEs!) ---
        time_begin=tic;
        if flag == 1
           fitnessU = my_rbfpredict(srgtSRGT.RBF_Model, srgtSRGT.P, U); % RBF预测
        else
           fitnessU  = srgtsPRSEvaluate(U, srgtSRGT);                    % PRS预测
        end
        time_cost=toc(time_begin);

        NFEs=NFEs+NP;                       % 代理模型评估次数累加

        %% --- 贪心选择: 子代优于父代则替换 ---
        for i=1:NP
            if(fitnessU(i)<fitnessP(i))     % 如果子代的代理模型预测值更好(更小)
                P(i,:)=U(i,:);              % 用子代替换父代
                fitnessP(i)=fitnessU(i);
                if(fitnessU(i)<fitnessBestP) % 如果是全局最优
                   fitnessBestP=fitnessU(i);
                   bestP_position = i;
                   bestP=U(i,:);            % 更新全局最优解
                end
            end
            recRMSE(NFEs)=fitnessP(i);
        end

        %% --- 提前停止判断 ---
        % 如果连续10代最优值改进量小于 minerror(=1e-20), 认为已收敛, 提前退出
        error=abs(fitnessBestP_old-fitnessBestP);
        if error <= minerror
            flag_er=flag_er+1;              % 连续未改进计数+1
        else
            flag_er=0;                      % 有改进则重置计数
        end
        if flag_er >=10                     % 连续10代无显著改进
            break;                          % 提前退出DE优化循环
        end
        G=G+1;                              % 代数+1
    end
    bestFitness=fitnessBestP;               % 返回代理模型上的最优预测值
    endNFEs = NFEs;
    time_cost=toc(time_begin);
end
