# V3 Monitor Summary

- Timestamp: `2026-04-14T17:29:52`

## GPU
- Available: `True`
- Query: `0, NVIDIA GeForce RTX 3090, 21804, 24576, 100`
- Compute apps: `2606330, python, 0
2626969, python, 0
2627091, python, 0
2627256, python, 0`

## Progress
- `atsp`: done `243/243`
- `mvrp`: done `780/1530`

## Running
```text
2606484       15:55 bash -lc source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh && conda activate easynco_zhoucl && export PYTHONPATH=/public/home/zhoucl/shiys && cd /public/home/zhoucl/shiys && python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mtpomo_part1.json --continue-on-error |& tee /public/home/zhoucl/shiys/easynco_v3_bridge/logs/v3_mvrp_mtpomo_part1.tmux.log
2606491       15:54 bash -lc source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh && conda activate easynco_zhoucl && export PYTHONPATH=/public/home/zhoucl/shiys && cd /public/home/zhoucl/shiys && python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mvmoe_part1.json --continue-on-error |& tee /public/home/zhoucl/shiys/easynco_v3_bridge/logs/v3_mvrp_mvmoe_part1.tmux.log
2606495       15:54 bash -lc source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh && conda activate easynco_zhoucl && export PYTHONPATH=/public/home/zhoucl/shiys && cd /public/home/zhoucl/shiys && python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mtpomo_part2.json --continue-on-error |& tee /public/home/zhoucl/shiys/easynco_v3_bridge/logs/v3_mvrp_mtpomo_part2.tmux.log
2606503       15:54 bash -lc source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh && conda activate easynco_zhoucl && export PYTHONPATH=/public/home/zhoucl/shiys && cd /public/home/zhoucl/shiys && python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mvmoe_part2.json --continue-on-error |& tee /public/home/zhoucl/shiys/easynco_v3_bridge/logs/v3_mvrp_mvmoe_part2.tmux.log
2607583       15:54 python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mtpomo_part1.json --continue-on-error
2607585       15:54 python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mtpomo_part2.json --continue-on-error
2607587       15:54 python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mvmoe_part1.json --continue-on-error
2607589       15:54 python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mvmoe_part2.json --continue-on-error
2626969       00:24 python eval.py settings=mvmoe_settings mode=test model=mvmoe problem=VRPBTW scale=91 batch_size=64 episodes=196 decoder_strategy=greedy cuda=[0] test_data_path=offline_init_v3/mvrp/vrpbtw/vrpbtw91_nums196.pkl settings.test_loader.model_dirpath=pretrained/mvmoe/mvmoe settings.test_loader.model_filename=mvmoe_mvrp100.ckpt settings.env.mode=1 ++settings.module.test_data_params.mode=1 dir=results/label_v3/mvmoe_vrpbtw/scale_91
2627091       00:19 python eval.py settings=mtpomo_settings mode=test model=mtpomo problem=VRPBTW scale=94 batch_size=64 episodes=196 decoder_strategy=greedy cuda=[0] test_data_path=offline_init_v3/mvrp/vrpbtw/vrpbtw94_nums196.pkl settings.test_loader.model_dirpath=pretrained/mtpomo/mtpomo settings.test_loader.model_filename=mtpomo_mvrp_100.ckpt settings.env.mode=1 ++settings.module.test_data_params.mode=1 dir=results/label_v3/mtpomo_vrpbtw/scale_94
2627256       00:10 python eval.py settings=mvmoe_settings mode=test model=mvmoe problem=OVRPL scale=86 batch_size=64 episodes=196 decoder_strategy=greedy cuda=[0] test_data_path=offline_init_v3/mvrp/ovrpl/ovrpl86_nums196.pkl settings.test_loader.model_dirpath=pretrained/mvmoe/mvmoe settings.test_loader.model_filename=mvmoe_mvrp100.ckpt settings.env.mode=1 ++settings.module.test_data_params.mode=1 dir=results/label_v3/mvmoe_ovrpl/scale_86
2627384       00:01 python eval.py settings=mtpomo_settings mode=test model=mtpomo problem=OVRPL scale=96 batch_size=64 episodes=196 decoder_strategy=greedy cuda=[0] test_data_path=offline_init_v3/mvrp/ovrpl/ovrpl96_nums196.pkl settings.test_loader.model_dirpath=pretrained/mtpomo/mtpomo settings.test_loader.model_filename=mtpomo_mvrp_100.ckpt settings.env.mode=1 ++settings.module.test_data_params.mode=1 dir=results/label_v3/mtpomo_ovrpl/scale_96
```

## Latest Activity
- `label_v3` latest write: `2026-04-14T17:29:53`
  path: `/public/home/zhoucl/shiys/EasyNCO/results/label_v3/mtpomo_vrpbtw/scale_94/eval.log`

## Recent Errors
- none
