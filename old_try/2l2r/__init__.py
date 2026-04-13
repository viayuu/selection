"""
2l2r: 基于 supervised learning-to-rank 的初始化方法选择。

当前模块的设计目标是：
1. 继续复用 2cmab2 已经稳定的数据协议
2. 继续使用同一套 NSS 风格图编码器
3. 把训练范式从 bandit 更新改成监督式排序学习
"""

