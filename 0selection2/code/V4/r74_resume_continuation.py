"""Exact accepted-clock R74 continuation8 using the immutable node-local mirror."""

import csv
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path

from .r74_train_only_continuation import exclusive, now, read


WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'code/V4/runs/R74_adaptive_incomplete_acquisition'
RECORD = ROOT / 'continuation8'
RUN = ROOT / 'B_adaptive_seed2'
GPU_UUID = 'GPU-cb626bdb-4319-2f5e-b91f-17fcaa44a83d'


def launch_identity():
    if (os.environ.get('SLURM_JOB_ID') != '1465' or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID
            or socket.gethostname().split('.')[0] != 'gpu03'):
        raise ValueError('Only existing1465/gpu03/assigned UUID may resume the formal model')
    cut, terminal = read(RECORD / 'checkpoint_cut.json'), read(RECORD / 'cut_terminal_receipt.json')
    if (not cut['safe_before_next_update'] or terminal['step'] != '1465.118'
            or not terminal['owned_tmux_gone'] or not terminal['actual_terminal']):
        raise ValueError('Require the exact persisted cut and actual prior owned-step terminal')
    if (RECORD / 'formal_started.json').exists() or (RECORD / 'failures.jsonl').exists():
        raise ValueError('Preserve the original exact-resume attempt; no automatic retry')
    while True:
        command = ['nvidia-smi', '--id='+GPU_UUID, '--query-gpu=index,uuid,memory.used,memory.total', '--format=csv,noheader,nounits']
        result = subprocess.run(command, text=True, capture_output=True)
        rows = list(csv.reader(result.stdout.splitlines()))
        if result.returncode or len(rows) != 1:
            raise RuntimeError('Assigned-only launch observation failed')
        index, uuid, used, total = [value.strip() for value in rows[0]]
        if index != '1' or uuid != GPU_UUID:
            raise ValueError('Assigned UUID/index mismatch')
        observation = dict(utc=now(), command=command, returncode=result.returncode,
                           stdout=result.stdout, stderr=result.stderr, idle=int(used)<500)
        with (RECORD / 'availability.jsonl').open('a') as stream:
            stream.write(json.dumps(observation)+'\n')
        if observation['idle']:
            break
        time.sleep(30)
    previous = read(ROOT / 'continuation7/trainer_launch.json')
    launch = dict(started_utc=now(), host=socket.gethostname(), trainer_pid=os.getpid(),
        trainer_ppid=os.getppid(), slurm_step='1465.'+os.environ['SLURM_STEP_ID'], gpu_uuid=GPU_UUID,
        gpu_index=1, tmux_session='r74_continuation8', tmux_pane=os.environ['R74_TMUX_PANE_ID'],
        tmux_pane_pid=os.environ['R74_TMUX_PANE_PID'], tmux_server_host=os.environ['R74_TMUX_SERVER_HOST'],
        command='bash code/V4/run_v4_r74_continuation8.sh', status='running', automatic_restart=False,
        execution_continuation=True, train_only_continuation=True, exact_checkpoint_resume=True,
        continuation_record='continuation8', prior_steps=previous['prior_steps'],
        charged_prior_Slurm_wall_seconds=3839, intentional_training_cut=terminal,
        preceding_accepted_training_step=previous['slurm_step'],
        resumed_from_successful_update=cut['last_update']['successful_update'],
        partial_epoch_timed_seconds=cut['partial_epoch_timed_seconds'],
        local_feature_mirror=os.environ['R74_LOCAL_FEATURE_MIRROR'])
    exclusive(RECORD / 'formal_started.json', launch)
    from .r73_frozen_evaluation import write
    write(RECORD / 'trainer_launch.json', launch)
    return launch


def main():
    started, launch, wb = time.monotonic(), None, None
    try:
        launch = launch_identity()
        import torch
        from . import r74_experiment as experiment
        from .r74_model import R74Model
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        protocol, generation = read(ROOT / 'protocol.json'), read(ROOT / 'feature_cache_complete.json')
        mirror = read(Path(os.environ['R74_LOCAL_FEATURE_MIRROR']))
        if (not mirror['complete'] or mirror['original_manifest'] != str(ROOT / 'feature_cache_complete.json')
                or not read(ROOT / 'probes_released.json')['no_live_probe_states']):
            raise ValueError('Reuse complete original generation and released probes only')
        schedule = experiment.paired_triples({task:10000 for task in experiment.PROBLEMS}, epochs=80, seed=2, batch_size=256)
        manifest, raw, caches = read(ROOT / 'data_manifest.json'), {}, {}
        for split, count in (('train',10000), ('val',1000)):
            raw[split], caches[split] = {}, {}
            for problem in experiment.PROBLEMS:
                value = experiment.read_raw_costs(problem, split)
                metadata = generation['per_task'][split+'/'+problem]
                if (not metadata['complete'] or metadata['rows'] != count or len(value['winner']) != count
                        or {key:value[key] for key in ('pool','pool_ids','data_hash','label_hash')} != manifest[split][problem]
                        or metadata['source_identity'] != manifest[split][problem]):
                    raise ValueError('Original complete native supervision and immutable feature identities required')
                raw[split][problem] = value
                caches[split][problem] = experiment.CachedTask(metadata, value)
        checkpoint = torch.load(RECORD / 'resume.pt', map_location='cpu', weights_only=False)
        cut = read(RECORD / 'checkpoint_cut.json')
        if (checkpoint['successful_updates'] != cut['last_update']['successful_update']
                or checkpoint['parameter_mode'] != 'averaged x' or checkpoint['checkpoint_mode'] != 'eval'
                or json.loads(json.dumps(checkpoint['args'])) != protocol):
            raise ValueError('Exactly the immutable complete naturally persisted cut may resume')
        for row in [json.loads(line) for line in (RUN/'updates.jsonl').read_text().splitlines() if line.strip()]:
            entries = experiment.triple_indices(schedule, row['global_epoch']-1, row['update_in_epoch']-1)
            if row['task_indices'] != {problem:indices.tolist() for problem,indices in entries}:
                raise ValueError('The accepted log must preserve the original exact task/row schedule')
        model = R74Model(protocol['model_params'])
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to('cuda:0').float().eval()
        counts = {key:value for key,value in read(ROOT/'continuation7/initialization.json').items()
                  if key in ('full_model_parameters','frozen_base_parameters','trainable_policy_parameters')}
        if (sum(p.numel() for p in model.parameters()) != counts['full_model_parameters']
                or sum(p.numel() for p in model.policy.parameters()) != counts['trainable_policy_parameters']
                or any(p.requires_grad for p in model.base.parameters())):
            raise ValueError('The original complete policy and frozen F must be strictly restored')
        for field in ('mean','std'):
            if not torch.equal(getattr(model.base.instance_encoder.geometry_residual,field),
                               torch.tensor(protocol['geometry_stats'][field],device='cuda:0',dtype=torch.float32)):
                raise ValueError('Original frozen normalizer buffers changed')
        experiment.atomic_json(RECORD/'initialization.json', dict(utc=now(), strict_complete_restore=True,
            checkpoint_loads=1, checkpoint=str(RECORD/'resume.pt'), parameter_mode='averaged x',
            resumed_from_successful_update=checkpoint['successful_updates'], fresh_optimizer_scaler_clock=False,
            generation_regenerated=False, base_or_probe_forwards=0, **counts))
        import wandb
        wb = wandb.init(**protocol['wandb'], name='R74_B_exact_resume8_seed2', dir=str(ROOT),
                        id='r74-b-cont8-'+launch['slurm_step'].replace('.','-'), config=protocol)
        experiment.train(protocol, schedule, model, caches, raw, generation, launch, wb, 'cuda:0', resume=checkpoint)
        elapsed = time.monotonic()-started
        launch.update(status='continuation_complete', completed_utc=now(), continuation_formal_wall_seconds=elapsed,
            cumulative_formal_wall_seconds=3839+launch['intentional_training_cut']['elapsed_seconds']+elapsed,
            generation_regenerated=False, new_base_or_probe_forwards=0)
        experiment.atomic_json(RECORD/'trainer_launch.json',launch)
    except BaseException as error:
        failure=dict(utc=now(),pid=os.getpid(),error=str(error),error_type=type(error).__name__,
                     traceback=traceback.format_exc(),automatic_restart=False,
                     slurm_step=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID',''),
                     continuation_formal_wall_seconds=time.monotonic()-started)
        with (RECORD/'failures.jsonl').open('a') as stream:
            stream.write(json.dumps(failure)+'\n')
        from .r73_frozen_evaluation import write
        write(RECORD/'pipeline_status.json',dict(failure,stage='failed_preserved_no_retry'))
        if launch is not None:
            launch.update(status='failed_preserved_no_retry',failed_utc=now())
            write(RECORD/'trainer_launch.json',launch)
        raise
    finally:
        if wb is not None:
            wb.finish()


if __name__ == '__main__':
    main()
