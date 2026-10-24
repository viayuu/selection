# R108 runner handoff

入口与已验证R107私有R84引擎一致：train、test、timing、describe、status。两组A_prior_ce/B_density_ratio；默认root为本目录，prepared-root仍为R84新90k的prepared，cache为原节点热缓存。

分类先验数据位于本目录`prepared/priors.json`，仅由冻结TRAIN计数生成。`initialize_shared`加载它作为固定模型buffer；checkpoint factory从state恢复，不读取待测标签、不重新拟合统计。

逻辑batch1024，初拟micro512/1024、eval256；源代码只在正式ready之前依据必要显存见证共同锁定。模型、prior、CLI和代码检查完成前不得声称已训练。R107和R106的任何源码不被修改。

W&B保留既有显式offline约定；训练完按完整VAL唯一权重做TEST/timing/LIB，再CPU PDF和整版GitHub发布。没有网络重试、偷偷续训或逐题最佳模型拼接。
