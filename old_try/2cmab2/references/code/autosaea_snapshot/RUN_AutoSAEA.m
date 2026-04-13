% =========================================================================
% RUN_AutoSAEA: 实验运行入口函数
%
% 作用: 对指定的测试函数多次运行AutoSAEA, 收集统计结果(均值/中位数/标准差)
%
% 输入:
%   runs   - 独立运行次数(每次运行有不同的随机种子, 用于统计分析)
%   D      - 决策变量维度
%   FUN    - 目标函数句柄 (如 @CEC05_f1)
%   LB     - 决策变量下界向量 (1×D)
%   UB     - 决策变量上界向量 (1×D)
%   fname  - 函数名称字符串 (用于显示和保存文件名)
%   f_bias - 函数的已知最优值偏移量 (用于计算误差 = 实际值 - 最优值)
%
% 输出:
%   gsamp1    - 每次运行的收敛曲线矩阵 (runs × MaxFEs)
%              gsamp1(r, t) = 第r次运行在第t次评估时的全局最优值
%   time_cost - 总运行时间(秒)
% =========================================================================
function [ gsamp1 ,time_cost] = RUN_AutoSAEA(runs, D, FUN, LB, UB, fname,f_bias)
time_begin=tic;                              % 记录总计时起点
warning('off');                              % 关闭所有警告(避免GP训练等产生大量警告)
addpath(genpath(pwd));                       % 将当前目录及所有子目录加入MATLAB路径
                                             % 确保所有函数文件都能被找到

%% --- 多次独立运行 ---
for r=1:runs
    fprintf('\n');
    disp(['FUNCTION: ', fname,' RUN: ', num2str(r)]);   % 显示当前运行信息
    fprintf('\n');

    % 调用AutoSAEA主函数, 进行一次完整的优化运行
    [hisf,mf,gfs]= AutoSAEA(FUN, D, LB, UB);
    % hisf: 所有历史适应度值
    % mf: MaxFEs (=1000)
    % gfs: 每次评估后的全局最优值 (1×MaxFEs)

    fprintf('Best fitness (PSO-final): %e\n',min(hisf));  % 打印本次运行的最终最优值
    gsamp1(r,:)=gfs(1:mf);                  % 保存本次运行的收敛曲线
end

%% =================== 统计结果输出 ===================
samp_mean   = mean(gsamp1(:,end));           % 所有运行最终最优值的均值
samp_mean_error = samp_mean - f_bias;        % 平均误差 = 均值 - 已知最优值
samp_median = median(gsamp1(:,end));         % 中位数
std_samp    = std(gsamp1(:,end));            % 标准差
gsamp1_ave  = mean(gsamp1,1);               % 所有运行的平均收敛曲线

%% --- 保存结果 ---
time_cost=toc(time_begin);                   % 计算总运行时间
save(strcat('result/NFE',num2str(mf),'_',fname,' runs=',num2str(runs),' Dim=',num2str(D)));
% 保存为.mat文件, 文件名包含: NFE数、函数名、运行次数、维度
end
