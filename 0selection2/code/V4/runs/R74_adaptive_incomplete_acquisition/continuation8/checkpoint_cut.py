"""Stop the owned trainer only inside its existing durable checkpoint boundary."""

import ctypes
import json
import os
import select
import shutil
import signal
import struct
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT=Path('/public/home/shiys/0selection2/code/V4/runs/R74_adaptive_incomplete_acquisition')
RECORD=ROOT/'continuation8'
RUN=ROOT/'B_adaptive_seed2'
PID=101584


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def main():
    if (not read(RECORD/'mirror_manifest.json')['complete']
            or not read(RECORD/'syntax_check.json')['returncode']==0
            or not read(ROOT/'delivery_refresh3/observer2_intentional_stop.json')['owned_tmux_gone']
            or (RECORD/'checkpoint_cut.json').exists()):
        raise ValueError('Prepare mirror/syntax and stop only observer2 before the exclusive safe cut')
    environment=dict(item.split(b'=',1) for item in Path(f'/proc/{PID}/environ').read_bytes().split(b'\0') if b'=' in item)
    if environment.get(b'SLURM_JOB_ID') != b'1465' or environment.get(b'SLURM_STEP_ID') != b'118':
        raise ValueError('The exact original owned trainer identity must still match')
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC)
    if fd<0 or libc.inotify_add_watch(fd,os.fsencode(RUN),0x80)<0:
        raise OSError(ctypes.get_errno(),'Cannot watch the existing atomic checkpoint rename')
    with (RECORD/'cut_watch_started.json').open('x') as stream:
        json.dump(dict(utc=now(),pid=os.getpid(),trainer_pid=PID,step='1465.118',
                       event='existing last.pt atomic rename; no new checkpoint save'),stream,indent=2)
    while True:
        select.select([fd],[],[])
        events=os.read(fd,65536)
        offset=0
        while offset<len(events):
            _,mask,_,length=struct.unpack_from('iIII',events,offset)
            name=events[offset+16:offset+16+length].rstrip(b'\0')
            offset+=16+length
            if not(mask&0x80) or name!=b'last.pt':
                continue
            os.kill(PID,signal.SIGSTOP)
            paused_utc=now()
            time.sleep(.02)
            rows=[json.loads(line) for line in (RUN/'updates.jsonl').read_text().splitlines() if line.strip()]
            last=rows[-1]
            status=read(ROOT/'continuation7/current_status.json')
            status_clock=status.get('last_metrics',{}).get('successful_update',0)
            safe=(last['successful_update']%60==0 and status_clock<last['successful_update']
                  and [row['successful_update'] for row in rows]==list(range(1,last['successful_update']+1)))
            if not safe:
                with (RECORD/'declined_cut_observations.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(utc=paused_utc,last_update=last['successful_update'],
                        published_status_clock=status_clock,reason='Cannot prove still inside save/status boundary; continued unchanged'))+'\n')
                os.kill(PID,signal.SIGCONT)
                continue
            with (RUN/'last.pt').open('rb') as source, (RECORD/'resume.pt').open('xb') as destination:
                shutil.copyfileobj(source,destination,1024*1024)
            (RECORD/'resume.pt').chmod(0o444)
            if (RUN/'last.pt').stat().st_size!=(RECORD/'resume.pt').stat().st_size:
                raise ValueError('The exclusive naturally persisted checkpoint copy is incomplete')
            receipt=dict(utc=now(),paused_utc=paused_utc,trainer_pid=PID,step='1465.118',
                safe_before_next_update=True,published_status_clock=status_clock,last_update=last,
                partial_epoch_timed_seconds=last['wall_seconds'] if last['epoch']==1 else sum(
                    row['seconds'] for row in rows if row['epoch']==last['epoch']),
                accepted_log_rows=len(rows),checkpoint_bytes=(RECORD/'resume.pt').stat().st_size,
                checkpoint_source=str(RUN/'last.pt'),checkpoint_copy=str(RECORD/'resume.pt'),
                checkpoint_content_loaded=False,extra_save_or_mode_transition=False,
                proof='last.pt rename follows complete optimizer update and averaged-x serialization; older published current_status proves the trainer has not passed the same cadence status call, which precedes any next update',
                prior_training_launch=read(ROOT/'continuation7/trainer_launch.json'),
                stopped_trainer_state=Path(f'/proc/{PID}/status').read_text().split('State:',1)[1].splitlines()[0].strip())
            with (RECORD/'checkpoint_cut.json').open('x') as stream:
                json.dump(receipt,stream,indent=2,allow_nan=False)
                stream.write('\n')
            print(json.dumps({key:value for key,value in receipt.items() if key not in ('last_update','prior_training_launch')}),flush=True)
            return


if __name__=='__main__':
    main()
