# 2cmab2 运行结果复盘

## 基本信息
- run 目录: `/mnt/d/Study/3/neural-solver-selection/2cmab2/outputs/2026-03-23-23-07-19_neural_linucb`
- method: `neural_linucb`
- reward_mode: `linear_zero_one`
- train/val/test: `14000 / 3000 / 3000`
- alpha: `1.0`, hidden_dim: `64`, train_every: `50`, representation_steps: `10`
- representation_buffer_size: `5000`, linear_head_buffer_size: `-1`
- freeze_encoder: `False`
- checkpoint 文件数: `0`
- 训练耗时: `5h 1m 44s`

## Selector 指标
| source | split | scope | count | mean_reward | top1_accuracy | mean_regret | mean_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selector | val_greedy | cvrp | 1500 | 0.832000 | 0.362667 | 0.084903 | 15.856593 |
| selector | val_greedy | tsp | 1500 | 0.852758 | 0.262000 | 0.010122 | 7.775747 |
| selector | test_greedy | cvrp | 1500 | 0.831429 | 0.358667 | 0.087679 | 15.850527 |
| selector | test_greedy | tsp | 1500 | 0.850515 | 0.248000 | 0.009892 | 7.771475 |
| selector | val_ucb | cvrp | 1500 | 0.832000 | 0.360667 | 0.084822 | 15.856513 |
| selector | val_ucb | tsp | 1500 | 0.851242 | 0.268667 | 0.010665 | 7.776290 |
| selector | test_ucb | cvrp | 1500 | 0.831143 | 0.352667 | 0.091307 | 15.854155 |
| selector | test_ucb | tsp | 1500 | 0.853485 | 0.262667 | 0.009755 | 7.771337 |

## Baseline 指标
| source | split | scope | count | mean_reward | top1_accuracy | mean_regret | mean_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | test_greedy | cvrp | 1500 | 0.492952 | 0.124000 | 2.285205 | 18.048054 |
| random | test_greedy | tsp | 1500 | 0.504788 | 0.088000 | 3.696003 | 11.457586 |
| single_best_global | test_greedy | cvrp | 1500 | 0.831619 | 0.411333 | 0.090761 | 15.853609 |
| single_best_global | test_greedy | tsp | 1500 | 0.846273 | 0.139333 | 0.010797 | 7.772379 |
| single_best_per_problem | test_greedy | cvrp | 1500 | 0.831619 | 0.411333 | 0.090761 | 15.853609 |
| single_best_per_problem | test_greedy | tsp | 1500 | 0.863727 | 0.300667 | 0.009191 | 7.770773 |
| oracle | test_greedy | cvrp | 1500 | 1 | 1 | 0 | 15.762848 |
| oracle | test_greedy | tsp | 1500 | 0.981485 | 1 | 0 | 7.761582 |

## Top-k 命中分析

- 这里的 `top1/top2/top3` 不是 summary.json 里原生给出的指标，而是根据 trace 中的 `selected_reward` 反推选中方法的 tie-aware 平均 rank 后得到。
- 因此它更适合回答“选中的方法是不是至少落在真实前 1 / 前 2 / 前 3 名”。
- 它和 `top1_accuracy` 的定义不同：`top1_accuracy` 用的是 `selected_arm == best_arm`。

### selector_train_ucb.jsonl
| scope | count | mean_selected_rank | top1_hit_rate | top2_hit_rate | top3_hit_rate |
| --- | --- | --- | --- | --- | --- |
| cvrp | 7000 | 2.192000 | 0.348857 | 0.619143 | 0.852571 |
| tsp | 7000 | 2.637286 | 0.189143 | 0.504571 | 0.719571 |

### selector_val_greedy.jsonl
| scope | count | mean_selected_rank | top1_hit_rate | top2_hit_rate | top3_hit_rate |
| --- | --- | --- | --- | --- | --- |
| cvrp | 1500 | 2.176000 | 0.362667 | 0.614667 | 0.854000 |
| tsp | 1500 | 2.619667 | 0.191333 | 0.500667 | 0.722000 |

### selector_val_ucb.jsonl
| scope | count | mean_selected_rank | top1_hit_rate | top2_hit_rate | top3_hit_rate |
| --- | --- | --- | --- | --- | --- |
| cvrp | 1500 | 2.176000 | 0.360667 | 0.612667 | 0.855333 |
| tsp | 1500 | 2.636333 | 0.188000 | 0.500000 | 0.714667 |

### selector_test_greedy.jsonl
| scope | count | mean_selected_rank | top1_hit_rate | top2_hit_rate | top3_hit_rate |
| --- | --- | --- | --- | --- | --- |
| cvrp | 1500 | 2.180000 | 0.358667 | 0.611333 | 0.858667 |
| tsp | 1500 | 2.644333 | 0.184000 | 0.487333 | 0.712667 |

### selector_test_ucb.jsonl
| scope | count | mean_selected_rank | top1_hit_rate | top2_hit_rate | top3_hit_rate |
| --- | --- | --- | --- | --- | --- |
| cvrp | 1500 | 2.182000 | 0.352667 | 0.618667 | 0.858667 |
| tsp | 1500 | 2.611667 | 0.195333 | 0.500667 | 0.716000 |

## 选臂分布
### selector_train_ucb.jsonl 选择集中度
| scope | count | unique_selected_arms | entropy | effective_arms |
| --- | --- | --- | --- | --- |
| tsp | 7000 | 11 | 1.192955 | 3.296809 |
| cvrp | 7000 | 8 | 1.014897 | 2.759079 |

### selector_val_greedy.jsonl 选择集中度
| scope | count | unique_selected_arms | entropy | effective_arms |
| --- | --- | --- | --- | --- |
| cvrp | 1500 | 4 | 1.027027 | 2.792751 |
| tsp | 1500 | 4 | 1.170366 | 3.223173 |

### selector_val_ucb.jsonl 选择集中度
| scope | count | unique_selected_arms | entropy | effective_arms |
| --- | --- | --- | --- | --- |
| cvrp | 1500 | 4 | 1.044016 | 2.840602 |
| tsp | 1500 | 5 | 1.153562 | 3.169462 |

### selector_test_greedy.jsonl 选择集中度
| scope | count | unique_selected_arms | entropy | effective_arms |
| --- | --- | --- | --- | --- |
| cvrp | 1500 | 3 | 0.998870 | 2.715212 |
| tsp | 1500 | 4 | 1.158811 | 3.186143 |

### selector_test_ucb.jsonl 选择集中度
| scope | count | unique_selected_arms | entropy | effective_arms |
| --- | --- | --- | --- | --- |
| cvrp | 1500 | 5 | 1.016877 | 2.764548 |
| tsp | 1500 | 5 | 1.157207 | 3.181035 |

### selector_train_ucb.jsonl 各 arm 被选比例
| scope | arm | count | fraction |
| --- | --- | --- | --- |
| tsp | icam | 3036 | 0.433714 |
| tsp | pointerformer | 2305 | 0.329286 |
| tsp | lehd | 1388 | 0.198286 |
| tsp | elg | 245 | 0.035000 |
| tsp | dact | 9 | 0.001286 |
| tsp | omni | 8 | 0.001143 |
| tsp | udc | 4 | 0.000571 |
| tsp | invit | 2 | 0.000286 |
| tsp | difusco | 1 | 0.000143 |
| tsp | t2t | 1 | 0.000143 |
| tsp | glop | 1 | 0.000143 |
| cvrp | lehd | 3586 | 0.512286 |
| cvrp | icam | 2388 | 0.341143 |
| cvrp | elg | 994 | 0.142000 |
| cvrp | omni | 25 | 0.003571 |
| cvrp | dact | 4 | 0.000571 |
| cvrp | lih | 1 | 0.000143 |
| cvrp | udc | 1 | 0.000143 |
| cvrp | invit | 1 | 0.000143 |

### selector_val_greedy.jsonl 各 arm 被选比例
| scope | arm | count | fraction |
| --- | --- | --- | --- |
| cvrp | lehd | 774 | 0.516000 |
| cvrp | icam | 418 | 0.278667 |
| cvrp | elg | 307 | 0.204667 |
| cvrp | omni | 1 | 0.000667 |
| tsp | icam | 609 | 0.406000 |
| tsp | pointerformer | 549 | 0.366000 |
| tsp | lehd | 288 | 0.192000 |
| tsp | elg | 54 | 0.036000 |

### selector_val_ucb.jsonl 各 arm 被选比例
| scope | arm | count | fraction |
| --- | --- | --- | --- |
| cvrp | lehd | 719 | 0.479333 |
| cvrp | icam | 536 | 0.357333 |
| cvrp | elg | 235 | 0.156667 |
| cvrp | omni | 10 | 0.006667 |
| tsp | icam | 686 | 0.457333 |
| tsp | pointerformer | 506 | 0.337333 |
| tsp | lehd | 251 | 0.167333 |
| tsp | elg | 55 | 0.036667 |
| tsp | omni | 2 | 0.001333 |

### selector_test_greedy.jsonl 各 arm 被选比例
| scope | arm | count | fraction |
| --- | --- | --- | --- |
| cvrp | lehd | 792 | 0.528000 |
| cvrp | icam | 456 | 0.304000 |
| cvrp | elg | 252 | 0.168000 |
| tsp | icam | 617 | 0.411333 |
| tsp | pointerformer | 553 | 0.368667 |
| tsp | lehd | 281 | 0.187333 |
| tsp | elg | 49 | 0.032667 |

### selector_test_ucb.jsonl 各 arm 被选比例
| scope | arm | count | fraction |
| --- | --- | --- | --- |
| cvrp | lehd | 729 | 0.486000 |
| cvrp | icam | 572 | 0.381333 |
| cvrp | elg | 188 | 0.125333 |
| cvrp | omni | 10 | 0.006667 |
| cvrp | dact | 1 | 0.000667 |
| tsp | icam | 686 | 0.457333 |
| tsp | pointerformer | 497 | 0.331333 |
| tsp | lehd | 258 | 0.172000 |
| tsp | elg | 58 | 0.038667 |
| tsp | omni | 1 | 0.000667 |

## 每个 arm 被选中后的真实效果
### selector_train_ucb.jsonl
| scope | arm | count | mean_reward | mean_regret | top1_rate | mean_cost |
| --- | --- | --- | --- | --- | --- | --- |
| cvrp | lehd | 3586 | 0.836706 | 0.087967 | 0.420524 | 15.858771 |
| cvrp | icam | 2388 | 0.827889 | 0.089144 | 0.286013 | 15.897115 |
| cvrp | elg | 994 | 0.817620 | 0.087760 | 0.251509 | 15.907691 |
| cvrp | omni | 25 | 0.651429 | 0.216963 | 0.040000 | 15.919853 |
| cvrp | dact | 4 | 0.214286 | 5.011737 | 0 | 20.282692 |
| cvrp | invit | 1 | 0.428571 | 0.970682 | 0 | 17.573004 |
| cvrp | lih | 1 | 0.285714 | 3.353895 | 0 | 19.415602 |
| cvrp | udc | 1 | 0 | 6.183142 | 0 | 21.233707 |
| tsp | icam | 3036 | 0.863352 | 0.009434 | 0.311924 | 7.786669 |
| tsp | pointerformer | 2305 | 0.856084 | 0.010220 | 0.142733 | 7.768408 |
| tsp | lehd | 1388 | 0.840451 | 0.010234 | 0.352305 | 7.762402 |
| tsp | elg | 245 | 0.769388 | 0.016984 | 0.097959 | 7.793648 |
| tsp | dact | 9 | 0.191919 | 1.789357 | 0 | 9.575404 |
| tsp | omni | 8 | 0.534091 | 0.127213 | 0 | 7.880438 |
| tsp | udc | 4 | 0.113636 | 2.318262 | 0 | 10.036646 |
| tsp | invit | 2 | 0.454545 | 0.173226 | 0 | 8.100258 |
| tsp | difusco | 1 | 0.636364 | 0.062112 | 0 | 7.700190 |
| tsp | glop | 1 | 0.363636 | 1.082900 | 0 | 8.386805 |
| tsp | t2t | 1 | 0.272727 | 0.897561 | 0 | 8.299824 |

### selector_val_greedy.jsonl
| scope | arm | count | mean_reward | mean_regret | top1_rate | mean_cost |
| --- | --- | --- | --- | --- | --- | --- |
| cvrp | lehd | 774 | 0.838686 | 0.086097 | 0.443152 | 15.820378 |
| cvrp | icam | 418 | 0.827409 | 0.081866 | 0.303828 | 15.886076 |
| cvrp | elg | 307 | 0.820847 | 0.086304 | 0.237785 | 15.896640 |
| cvrp | omni | 1 | 1 | 0 | 1 | 19.269119 |
| tsp | icam | 609 | 0.863562 | 0.008666 | 0.305419 | 7.778001 |
| tsp | pointerformer | 549 | 0.853453 | 0.011160 | 0.156648 | 7.771430 |
| tsp | lehd | 288 | 0.841067 | 0.009988 | 0.385417 | 7.776139 |
| tsp | elg | 54 | 0.786195 | 0.016703 | 0.185185 | 7.792126 |

### selector_val_ucb.jsonl
| scope | arm | count | mean_reward | mean_regret | top1_rate | mean_cost |
| --- | --- | --- | --- | --- | --- | --- |
| cvrp | lehd | 719 | 0.842043 | 0.083880 | 0.454798 | 15.806641 |
| cvrp | icam | 536 | 0.825693 | 0.084057 | 0.289179 | 15.857044 |
| cvrp | elg | 235 | 0.821277 | 0.084111 | 0.242553 | 15.964161 |
| cvrp | omni | 10 | 0.700000 | 0.210322 | 0.200000 | 16.884078 |
| tsp | icam | 686 | 0.864365 | 0.009070 | 0.317784 | 7.779684 |
| tsp | pointerformer | 506 | 0.848635 | 0.011539 | 0.154150 | 7.771621 |
| tsp | lehd | 251 | 0.840275 | 0.009696 | 0.398406 | 7.773709 |
| tsp | elg | 55 | 0.772727 | 0.021173 | 0.127273 | 7.784721 |
| tsp | omni | 2 | 0.545455 | 0.169151 | 0 | 7.885118 |

### selector_test_greedy.jsonl
| scope | arm | count | mean_reward | mean_regret | top1_rate | mean_cost |
| --- | --- | --- | --- | --- | --- | --- |
| cvrp | lehd | 792 | 0.834776 | 0.087069 | 0.422980 | 15.771133 |
| cvrp | icam | 456 | 0.833647 | 0.085923 | 0.302632 | 15.840186 |
| cvrp | elg | 252 | 0.816893 | 0.092775 | 0.257937 | 16.118765 |
| tsp | icam | 617 | 0.857743 | 0.009541 | 0.288493 | 7.768265 |
| tsp | pointerformer | 553 | 0.846293 | 0.011152 | 0.137432 | 7.776750 |
| tsp | lehd | 281 | 0.854254 | 0.007727 | 0.391459 | 7.772552 |
| tsp | elg | 49 | 0.785714 | 0.012517 | 0.163265 | 7.746177 |

### selector_test_ucb.jsonl
| scope | arm | count | mean_reward | mean_regret | top1_rate | mean_cost |
| --- | --- | --- | --- | --- | --- | --- |
| cvrp | lehd | 729 | 0.832843 | 0.086994 | 0.418381 | 15.775455 |
| cvrp | icam | 572 | 0.836414 | 0.086252 | 0.302448 | 15.904319 |
| cvrp | elg | 188 | 0.822188 | 0.090127 | 0.265957 | 15.965538 |
| cvrp | omni | 10 | 0.657143 | 0.170355 | 0.100000 | 16.058235 |
| cvrp | dact | 1 | 0 | 5.558678 | 0 | 21.551868 |
| tsp | icam | 686 | 0.861317 | 0.009039 | 0.303207 | 7.772203 |
| tsp | pointerformer | 497 | 0.849918 | 0.011059 | 0.146881 | 7.777198 |
| tsp | lehd | 258 | 0.854475 | 0.007230 | 0.391473 | 7.763043 |
| tsp | elg | 58 | 0.793887 | 0.015338 | 0.206897 | 7.744222 |
| tsp | omni | 1 | 0.454545 | 0.180790 | 0 | 7.977621 |

## test 集各方法 mean cost 与 selector 对比
### selector_test_greedy.jsonl 汇总
| scope | selector_mean_cost | n_methods | methods_better_than_selector | methods_worse_than_selector | methods_equal_to_selector | selector_rank_among_methods |
| --- | --- | --- | --- | --- | --- | --- |
| cvrp | 15.850527 | 8 | 1 | 7 | 0 | 2 |
| tsp | 7.771475 | 12 | 2 | 10 | 0 | 3 |

### selector_test_ucb.jsonl 汇总
| scope | selector_mean_cost | n_methods | methods_better_than_selector | methods_worse_than_selector | methods_equal_to_selector | selector_rank_among_methods |
| --- | --- | --- | --- | --- | --- | --- |
| cvrp | 15.854155 | 8 | 3 | 5 | 0 | 4 |
| tsp | 7.771337 | 12 | 2 | 10 | 0 | 3 |

### selector_test_greedy.jsonl 各方法 mean cost
| scope | arm | count | mean_cost | selector_mean_cost | delta_vs_selector | selector_better |
| --- | --- | --- | --- | --- | --- | --- |
| cvrp | icam | 1500 | 15.849489 | 15.850527 | -0.001038 | false |
| cvrp | lehd | 1500 | 15.853609 | 15.850527 | 0.003082 | true |
| cvrp | elg | 1500 | 15.853644 | 15.850527 | 0.003116 | true |
| cvrp | omni | 1500 | 15.963844 | 15.850527 | 0.113317 | true |
| cvrp | invit | 1500 | 16.353747 | 15.850527 | 0.503219 | true |
| cvrp | lih | 1500 | 20.541339 | 15.850527 | 4.690812 | true |
| cvrp | dact | 1500 | 20.848404 | 15.850527 | 4.997877 | true |
| cvrp | udc | 1500 | 22.648666 | 15.850527 | 6.798138 | true |
| tsp | lehd | 1500 | 7.769759 | 7.771475 | -0.001715 | false |
| tsp | icam | 1500 | 7.770773 | 7.771475 | -0.000702 | false |
| tsp | pointerformer | 1500 | 7.772379 | 7.771475 | 0.000905 | true |
| tsp | elg | 1500 | 7.778870 | 7.771475 | 0.007395 | true |
| tsp | omni | 1500 | 7.858207 | 7.771475 | 0.086733 | true |
| tsp | difusco | 1500 | 7.890429 | 7.771475 | 0.118954 | true |
| tsp | invit | 1500 | 7.977307 | 7.771475 | 0.205832 | true |
| tsp | glop | 1500 | 8.514799 | 7.771475 | 0.743324 | true |
| tsp | t2t | 1500 | 9.169899 | 7.771475 | 1.398424 | true |
| tsp | dact | 1500 | 9.682358 | 7.771475 | 1.910883 | true |
| tsp | udc | 1500 | 10.180803 | 7.771475 | 2.409328 | true |
| tsp | lih | 1500 | 49.078221 | 7.771475 | 41.306746 | true |

### selector_test_ucb.jsonl 各方法 mean cost
| scope | arm | count | mean_cost | selector_mean_cost | delta_vs_selector | selector_better |
| --- | --- | --- | --- | --- | --- | --- |
| cvrp | icam | 1500 | 15.849489 | 15.854155 | -0.004666 | false |
| cvrp | lehd | 1500 | 15.853609 | 15.854155 | -0.000546 | false |
| cvrp | elg | 1500 | 15.853644 | 15.854155 | -0.000512 | false |
| cvrp | omni | 1500 | 15.963844 | 15.854155 | 0.109689 | true |
| cvrp | invit | 1500 | 16.353747 | 15.854155 | 0.499592 | true |
| cvrp | lih | 1500 | 20.541339 | 15.854155 | 4.687184 | true |
| cvrp | dact | 1500 | 20.848404 | 15.854155 | 4.994249 | true |
| cvrp | udc | 1500 | 22.648666 | 15.854155 | 6.794510 | true |
| tsp | lehd | 1500 | 7.769759 | 7.771337 | -0.001578 | false |
| tsp | icam | 1500 | 7.770773 | 7.771337 | -0.000565 | false |
| tsp | pointerformer | 1500 | 7.772379 | 7.771337 | 0.001042 | true |
| tsp | elg | 1500 | 7.778870 | 7.771337 | 0.007533 | true |
| tsp | omni | 1500 | 7.858207 | 7.771337 | 0.086870 | true |
| tsp | difusco | 1500 | 7.890429 | 7.771337 | 0.119091 | true |
| tsp | invit | 1500 | 7.977307 | 7.771337 | 0.205969 | true |
| tsp | glop | 1500 | 8.514799 | 7.771337 | 0.743461 | true |
| tsp | t2t | 1500 | 9.169899 | 7.771337 | 1.398562 | true |
| tsp | dact | 1500 | 9.682358 | 7.771337 | 1.911021 | true |
| tsp | udc | 1500 | 10.180803 | 7.771337 | 2.409465 | true |
| tsp | lih | 1500 | 49.078221 | 7.771337 | 41.306883 | true |

## greedy 与 ucb 的差异
| scope | total | same_selection | different_selection | different_ratio | ucb_better_when_diff | ucb_worse_when_diff | equal_when_diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cvrp | 1500 | 1247 | 253 | 0.168667 | 125 | 128 | 0 |
| tsp | 1500 | 1153 | 347 | 0.231333 | 170 | 141 | 36 |

| uid | problem | greedy_arm | greedy_score | greedy_regret | ucb_arm | ucb_score | ucb_regret | best_arm | best_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cvrp:1010 | cvrp | lehd | 0.835365 | 0 | icam | 0.858946 | 0.288399 | lehd | 18.030581 |
| cvrp:102 | cvrp | elg | 0.845986 | 0.020197 | lehd | 0.913997 | 0.074341 | icam | 15.366686 |
| cvrp:1027 | cvrp | icam | 0.829254 | 0 | elg | 0.888826 | 0.059476 | icam | 13.003302 |
| cvrp:106 | cvrp | icam | 0.830977 | 0.057542 | lehd | 0.854237 | 0.051605 | omni | 13.589450 |
| cvrp:1079 | cvrp | elg | 0.780106 | 0.084965 | icam | 0.801819 | 0 | icam | 19.456673 |
| cvrp:1139 | cvrp | elg | 0.830138 | 0.042837 | lehd | 0.856113 | 0.085478 | icam | 18.019506 |
| cvrp:1243 | cvrp | lehd | 0.896666 | 0 | icam | 0.956898 | 0.027893 | lehd | 17.670795 |
| cvrp:1244 | cvrp | lehd | 0.904365 | 0 | icam | 0.952459 | 0.092880 | lehd | 15.188339 |
| cvrp:1288 | cvrp | lehd | 0.878678 | 0.204666 | icam | 0.931765 | 0.093544 | elg | 20.455551 |
| cvrp:1298 | cvrp | icam | 0.905134 | 0.187460 | elg | 0.977687 | 0 | elg | 18.091629 |
| cvrp:1320 | cvrp | elg | 0.863966 | 0 | icam | 0.888720 | 0.120375 | elg | 16.241957 |
| cvrp:1402 | cvrp | elg | 0.786149 | 0.265333 | lehd | 0.852896 | 0 | lehd | 16.202568 |
| cvrp:1493 | cvrp | elg | 0.704861 | 0.074207 | lehd | 0.806456 | 0.026867 | icam | 18.190746 |
| cvrp:1505 | cvrp | icam | 0.668678 | 0.006146 | omni | 0.736232 | 0 | omni | 13.581478 |
| cvrp:1530 | cvrp | lehd | 0.958406 | 0.106879 | icam | 1.001669 | 0 | icam | 14.160683 |
| cvrp:159 | cvrp | icam | 0.900736 | 0 | elg | 0.956971 | 0.008718 | icam | 16.615356 |
| cvrp:1611 | cvrp | elg | 0.831545 | 0 | icam | 0.854204 | 0.034554 | elg | 16.317635 |
| cvrp:1623 | cvrp | lehd | 0.826874 | 0 | icam | 0.857810 | 0.269540 | lehd | 15.257771 |
| cvrp:1683 | cvrp | lehd | 0.878823 | 0.130753 | icam | 0.937782 | 0.044527 | elg | 12.407223 |
| cvrp:1736 | cvrp | lehd | 0.798334 | 0.705536 | icam | 0.837960 | 0 | icam | 18.275806 |

## oracle 最优 arm 分布
| scope | arm | oracle_best_count | oracle_best_fraction |
| --- | --- | --- | --- |
| cvrp | lehd | 617 | 0.411333 |
| cvrp | icam | 440 | 0.293333 |
| cvrp | elg | 381 | 0.254000 |
| cvrp | omni | 62 | 0.041333 |
| tsp | lehd | 553 | 0.368667 |
| tsp | icam | 451 | 0.300667 |
| tsp | pointerformer | 209 | 0.139333 |
| tsp | elg | 201 | 0.134000 |
| tsp | difusco | 80 | 0.053333 |
| tsp | omni | 6 | 0.004000 |

## oracle 覆盖度
| arm | oracle_best_count | selected_count | selected_minus_oracle | coverage_over_oracle |
| --- | --- | --- | --- | --- |
| lehd | 1170 | 1073 | -97 | 0.917094 |
| icam | 891 | 1073 | 182 | 1.204265 |
| elg | 582 | 301 | -281 | 0.517182 |
| pointerformer | 209 | 553 | 344 | 2.645933 |
| difusco | 80 | 0 | -80 | 0 |
| omni | 68 | 0 | -68 | 0 |
