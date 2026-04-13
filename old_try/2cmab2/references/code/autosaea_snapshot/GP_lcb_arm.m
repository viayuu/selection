% =========================================================================
% GP_lcb_arm: 组合臂 {GP, LCB} 的执行函数
% 对应论文 IV.A节 第1)点 + 公式(2)-(6)(12)
%
% 工作流程:
%   1. 用当前种群训练GP(高斯过程)代理模型 (论文公式(2)-(6))
%   2. 对每个DE子代, GP预测其均值 f̂(x) 和标准差 ŝ(x)
%   3. 用LCB准则选候选解 (论文公式(12))
%   4. 真实评估 + 计算奖励
%
% LCB(Lower Confidence Bound)的直觉:
%   LCB(x) = f̂(x) - w·ŝ(x)
%   - f̂(x) 小: 预测值好 (利用)
%   - ŝ(x) 大: 不确定性高 (探索)
%   选 LCB 最小的点, 同时兼顾利用和探索
%
% 论文公式(12):
%   x* = argmin_{x∈X} ( f̂_{GP}(x) - w·ŝ(x) )
%   其中 w=2 (论文: "w is set to 2")
% =========================================================================
function [hx, hf, reward, NFEs, CE, gfs] = GP_lcb_arm(ghx, ghf, offspring, hx, hf, FUN, NFEs, CE, gfs)

 try  % try-catch: GP训练可能因数值问题(如矩阵奇异)失败, 失败时奖励为0
     theta =[];                             % GP超参数θ, 设为空则自动优化
                                            % 对应论文公式(4): C(x_i,x_j) = exp(-Σ θ_d·|x_{i,d}-x_{j,d}|²)

     %% --- 数据预处理: 去除重复样本 ---
     % GP训练时重复样本会导致协方差矩阵K奇异(不可逆)
     t_s = [ghx, ghf'];                     % 将决策变量和适应度拼接
     D = size(ghx, 2);                      % 维度D
     [t_s, ~,~] = unique(t_s, 'rows');      % 去除完全相同的行
     ghx = t_s(:, 1:D);                     % 去重后的训练输入
     ghf = t_s(:, D+1);                     % 去重后的训练输出

     %% --- 步骤1: 训练GP模型 ---
     % 对应论文公式(2)-(6), 使用DACE工具箱
     % @regpoly0: 零阶回归函数(常数趋势), 对应论文中 μ(x) 的全局趋势部分
     % @corrgauss: 高斯相关函数, 对应论文公式(4): C(x_i,x_j) = exp(-Σ θ_d·|...|²)
     % theta: 超参数, 通过最大化似然函数自动求解
     try
        [dmodel,~]=...
         GP_fit(ghx,ghf,@regpoly0,@corrgauss,theta);
     catch
        [dmodel,~]=...
        GP_fit(ghx,ghf,@regpoly0,@corrgauss,theta);
     end

     %% --- 步骤2+3: 计算每个子代的LCB值, 选最优的 ---
     w = 2;     % LCB权重参数 (论文: "w is set to 2 in this article")
                % w越大越倾向探索(更重视不确定性)
     for i = 1 : size(offspring, 1)          % 遍历每个DE子代
         % predictor 返回:
         %   tempobj: 预测均值 f̂_{GP}(x), 对应论文公式(5)
         %   MSE: 预测均方误差 ŝ²(x), 对应论文公式(6)
         [tempobj,~, MSE,~] =  predictor(offspring(i,:),dmodel);
         % LCB准则, 对应论文公式(12):
         % LCB(x) = f̂_{GP}(x) - w·ŝ(x)
         % 注意: ŝ(x) = sqrt(MSE), 即标准差
         OffObj(i) = tempobj - w * sqrt(MSE);
     end

     %% --- 选LCB最小的子代作为候选解 ---
     % 对应论文公式(12): x* = argmin_{x∈X} LCB(x)
     [~,I] = min(OffObj);
     candidate_position = offspring(I,:);     % 候选解 x_t

     %% --- 去重检查 + 真实评估 + 奖励计算 ---
     [~,ih,~] = intersect(hx,candidate_position,'rows');
     if isempty(ih)==1
        candidate_fit=FUN(candidate_position);  % 真实评估 (消耗1次FE)
        NFEs = NFEs + 1;
        hx=[hx; candidate_position];  hf=[hf, candidate_fit];  % 更新数据库

        CE(NFEs,:)=[NFEs,candidate_fit];
        gfs(1,NFEs)=min(CE(1:NFEs,2));

        Arm = num2str('GP_lcb ');
        [reward] = Low_level_r(ghf, hf, candidate_fit, NFEs, Arm);  % 低层奖励(公式24)
     else
        reward = 0;
     end
 catch
      reward = 0;   % GP训练失败(数值问题), 本轮不贡献评估, 奖励为0
 end

end
