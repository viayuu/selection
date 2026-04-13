% =========================================================================
% DEoperator: 差分进化(DE)变异和交叉算子
% 对应论文公式(14)-(16), 即 DE/current-to-best/1/bin 策略
%
% 作用: 从当前种群P出发, 生成N个新的子代解(offspring)
% 这些子代随后会被代理模型评估, 用于填充准则选择最优候选解
%
% 输入:
%   P    - 当前种群矩阵 (NP×Dim), 每行是一个D维解
%   NP   - 种群大小 N
%   Dim  - 决策变量维度 D
%   hisx - 历史最优解矩阵(已按适应度排序), hisx(1,:)是当前最优解 x_best
%   F    - DE缩放因子(论文设为0.5), 控制变异步长
%   CR   - DE交叉率(论文设为0.9), 控制从变异向量继承基因的概率
%   UB   - 变量上界矩阵 (NP×Dim)
%   LB   - 变量下界矩阵 (NP×Dim)
%
% 输出:
%   U    - 子代矩阵 (NP×Dim), 每行是一个新生成的候选解
% =========================================================================
function [U] = DEoperating(P,NP,Dim,hisx,F,CR,UB,LB)
    for i=1:NP  % 对种群中的每个个体 x_i 生成一个子代 o_i

        %% ========== 变异操作 (Mutation) ==========
        % 论文公式(14): v_i = x_i + F·(x_best - x_i) + F·(x_r1 - x_r2)
        % 这是 DE/current-to-best/1 变异策略:
        %   x_i:    当前个体(基向量)
        %   x_best: 当前种群最优个体(引导搜索方向)
        %   x_r1, x_r2: 随机选的两个不同个体(提供随机扰动)

        % --- 随机选择3个互不相同且不等于i的个体索引 ---
        k0=randi([1,NP]);           % 随机选第1个个体索引
        while(k0==i)                % 确保不等于当前个体i
            k0=randi([1,NP]);
        end
        P1=P(k0,:);                 % x_r1 (在公式中用作差分向量的一部分)

        k1=randi([1,NP]);           % 随机选第2个个体索引
        while(k1==i||k1==k0)        % 确保不等于i且不等于k0
            k1=randi([1,NP]);
        end
        P2=P(k1,:);                 % x_r1 → 代码中用作 P2 (对应公式中的 x_r1)

        k2=randi([1,NP]);           % 随机选第3个个体索引
        while(k2==i||k2==k1||k2==k0) % 确保3个索引互不相同且不等于i
            k2=randi([1,NP]);
        end
        P3=P(k2,:);                 % x_r2 → 代码中用作 P3 (对应公式中的 x_r2)

        % --- 执行变异 ---
        Xpbest = hisx(1,:);         % x_best: 当前种群的最优解(hisx已排序, 第1行最优)
        % 论文公式(14): v_i = x_i + F·(x_best - x_i) + F·(x_r1 - x_r2)
        V(i,:)=P(i,:)+F.*(Xpbest-P(i,:))+F.*(P2-P3);
        % 解读:
        %   F·(Xpbest - P(i,:)): 向最优解方向靠近(利用), F=0.5意味着走一半距离
        %   F·(P2 - P3):         随机方向扰动(探索)

        %% ========== 边界修复 (Boundary Repair) ==========
        % 对应论文公式(16): 如果 o_{i,j} 越界, 则随机重置到合法范围内
        % o_{i,j} = x_{l,j} + rand × (x_{u,j} - x_{l,j})
        for j=1:Dim
          if (V(i,j)>UB(i,j)||V(i,j)<LB(i,j))     % 如果第j维越界
             V(i,j)=LB(i,j)+rand*(UB(i,j)-LB(i,j)); % 随机重置到[LB, UB]范围内
          end
        end

        %% ========== 交叉操作 (Crossover) ==========
        % 对应论文公式(15): 二项交叉(binomial crossover)
        % o_{i,j} = v_{i,j}  如果 rand ≤ CR 或 j == j_rand
        %           x_{i,j}  否则
        jrand=randi([1,Dim]);       % j_rand: 随机选一个维度, 保证至少有一维来自变异向量
        for j=1:Dim
            k3=rand;                % 生成[0,1]随机数
            if(k3<=CR||j==jrand)    % 如果随机数 ≤ CR(=0.9) 或者是 j_rand 维
                U(i,j)=V(i,j);     % 从变异向量V继承 (90%概率)
            else
                U(i,j)=P(i,j);     % 从父代保留 (10%概率)
            end
        end
    end
end
