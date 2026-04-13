#!/bin/bash
set -euo pipefail

cd /public/home/zhoucl/shiys/EasyNCO
export PYTHONUNBUFFERED=1

/public/home/zhoucl/anaconda3/envs/easynco_zhoucl/bin/python run_nss_eval_suite.py --skip-lib-unstarted --exclude-methods difusco,t2t "$@"
