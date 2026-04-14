# Chat Live Status

- Timestamp: `2026-04-14T17:29:41`
- GPU: util 100%, mem 22121/24576 MiB
- `atsp`: `243/243` complete
- `mvrp`: `778/1530` complete

## Latest Writes
- `label_v3`: `2026-04-14T17:29:41`
  path: `/public/home/zhoucl/shiys/EasyNCO/results/label_v3/mtpomo_ovrpl/scale_95/eval.log`

## Running
```text
2606484       15:43 bash -lc source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh && conda activate easynco_zhoucl && export PYTHONPATH=/public/home/zhoucl/shiys && cd /public/home/zhoucl/shiys && python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mtpomo_part1.json --continue-on-error |& tee /public/home/zhoucl/shiys/easynco_v3_bridge/logs/v3_mvrp_mtpomo_part1.tmux.log
2606491       15:43 bash -lc source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh && conda activate easynco_zhoucl && export PYTHONPATH=/public/home/zhoucl/shiys && cd /public/home/zhoucl/shiys && python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mvmoe_part1.json --continue-on-error |& tee /public/home/zhoucl/shiys/easynco_v3_bridge/logs/v3_mvrp_mvmoe_part1.tmux.log
2606495       15:43 bash -lc source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh && conda activate easynco_zhoucl && export PYTHONPATH=/public/home/zhoucl/shiys && cd /public/home/zhoucl/shiys && python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mtpomo_part2.json --continue-on-error |& tee /public/home/zhoucl/shiys/easynco_v3_bridge/logs/v3_mvrp_mtpomo_part2.tmux.log
2606503       15:43 bash -lc source /public/home/zhoucl/anaconda3/etc/profile.d/conda.sh && conda activate easynco_zhoucl && export PYTHONPATH=/public/home/zhoucl/shiys && cd /public/home/zhoucl/shiys && python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mvmoe_part2.json --continue-on-error |& tee /public/home/zhoucl/shiys/easynco_v3_bridge/logs/v3_mvrp_mvmoe_part2.tmux.log
2607583       15:42 python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mtpomo_part1.json --continue-on-error
2607585       15:42 python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mtpomo_part2.json --continue-on-error
2607587       15:42 python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mvmoe_part1.json --continue-on-error
2607589       15:42 python /public/home/zhoucl/shiys/easynco_v3_bridge/scripts/run_label_queue.py --jobs /public/home/zhoucl/shiys/easynco_v3_bridge/manifests/mvrp_mvmoe_part2.json --continue-on-error
2626834       00:21 python eval.py settings=mvmoe_settings mode=test model=mvmoe problem=OVRPL scale=85 batch_size=64 episodes=196 decoder_strategy=greedy cuda=[0] test_data_path=offline_init_v3/mvrp/ovrpl/ovrpl85_nums196.pkl settings.test_loader.model_dirpath=pretrained/mvmoe/mvmoe settings.test_loader.model_filename=mvmoe_mvrp100.ckpt settings.env.mode=1 ++settings.module.test_data_params.mode=1 dir=results/label_v3/mvmoe_ovrpl/scale_85
2626969       00:12 python eval.py settings=mvmoe_settings mode=test model=mvmoe problem=VRPBTW scale=91 batch_size=64 episodes=196 decoder_strategy=greedy cuda=[0] test_data_path=offline_init_v3/mvrp/vrpbtw/vrpbtw91_nums196.pkl settings.test_loader.model_dirpath=pretrained/mvmoe/mvmoe settings.test_loader.model_filename=mvmoe_mvrp100.ckpt settings.env.mode=1 ++settings.module.test_data_params.mode=1 dir=results/label_v3/mvmoe_vrpbtw/scale_91
2627015       00:11 python eval.py settings=mtpomo_settings mode=test model=mtpomo problem=OVRPL scale=95 batch_size=64 episodes=196 decoder_strategy=greedy cuda=[0] test_data_path=offline_init_v3/mvrp/ovrpl/ovrpl95_nums196.pkl settings.test_loader.model_dirpath=pretrained/mtpomo/mtpomo settings.test_loader.model_filename=mtpomo_mvrp_100.ckpt settings.env.mode=1 ++settings.module.test_data_params.mode=1 dir=results/label_v3/mtpomo_ovrpl/scale_95
2627091       00:07 python eval.py settings=mtpomo_settings mode=test model=mtpomo problem=VRPBTW scale=94 batch_size=64 episodes=196 decoder_strategy=greedy cuda=[0] test_data_path=offline_init_v3/mvrp/vrpbtw/vrpbtw94_nums196.pkl settings.test_loader.model_dirpath=pretrained/mtpomo/mtpomo settings.test_loader.model_filename=mtpomo_mvrp_100.ckpt settings.env.mode=1 ++settings.module.test_data_params.mode=1 dir=results/label_v3/mtpomo_vrpbtw/scale_94
```

## Recent Errors
- none
