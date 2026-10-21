"""Small CPU-only R81 integration witnesses using synthetic data and temp roots."""

import csv
import io
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from code.unified_selector.registry import GLOBAL_SOLVERS, PROBLEMS
from . import r81_experiment as runner
from . import r81_reporting as reporting
from .method_auc_objective import AUC_NEG_KEY, AUC_POS_KEY, method_auc_targets, train_eligible_methods
from .r76_experiment import fixed_sbs_metrics, slice_batch, write_json
from .test_r76 import FakeCache, fixture_model


class ClosingCache(FakeCache):
    def close(self):
        pass


def synthetic_report_fixture(root):
    """All numbers below are fabricated TEST FIXTURES, never project results."""
    run = root / runner.GROUP
    run.mkdir(parents=True)
    runner.save_checkpoint(run / 'best.pt', {'synthetic_fixture': True})
    costs = np.asarray([[1.,2.,3.],[2.,1.,3.],[3.,2.,1.],[1.,1.,2.]],dtype=np.float64)
    scores = np.asarray([[2.,0.,1.],[1.,2.,0.],[0.,1.,2.],[2.,0.,1.]],dtype=np.float32)
    ids, pool = [0,1,2], GLOBAL_SOLVERS[:3]
    winner = costs.argmin(1).astype(np.int64)
    baseline = dict(sbs_solver=pool[0], sbs_slot=0, selection_split='train', train_rows=70114,
        mean_cost_by_solver=dict(zip(pool,costs.mean(0).tolist())))
    raw = dict(costs=costs,winner=winner,pool=pool,pool_ids=ids)
    metric, pred = fixed_sbs_metrics(scores,raw,baseline)
    baseline_map = {p:dict(baseline) for p in PROBLEMS}
    manifest = {s:{p:dict(rows=4,pool=pool,pool_ids=ids) for p in PROBLEMS} for s in ('train','val','test')}
    config = dict(experiment='R81',alias=runner.ALIAS,group=runner.GROUP,arm=runner.GROUP,
        data_root='/synthetic/R81_FIXTURE_NOT_REAL_DATA',baselines=baseline_map,data_manifest=manifest,
        max_successful_updates=runner.MAX_UPDATES,checkpoint_rule='strict VAL fixture rule',
        inference_rule='maximin, stable global-ID ties')
    state_hash = runner.json_hash(config)
    sha = runner.file_hash(run / 'best.pt')
    lock = dict(experiment='R81',group=runner.GROUP,arm=runner.GROUP,checkpoint=str(run/'best.pt'),
        checkpoint_sha256=sha,state_hash=state_hash,epoch=1,selected_update=1242,
        selected_without_test=True,per_problem_assembly=False,rule=config['checkpoint_rule'],inference_rule=config['inference_rule'])
    support, archives = {}, {}
    for split,tag in (('val','selected_best'),('test','selected_best')):
        directory = run / 'evaluation' / split / tag
        identity = dict(experiment='R81',group=runner.GROUP,split=split,successful_updates=1242,
            state_hash=state_hash,model_state_sha256='synthetic_state',checkpoint_sha256=sha if split=='test' else None)
        arrays = {}
        for p in PROBLEMS:
            values = dict(logits=scores,utility=scores.copy(),costs=costs,winner=winner,pred=pred,
                indices=np.arange(4),source_indices=np.arange(4),pool=np.asarray(pool),pool_ids=np.asarray(ids),nodes=np.full(4,5))
            arrays.update({p+'__'+key:value for key,value in values.items()})
            pos,neg,eligible = runner.native_targets(costs)
            support[p] = dict(pool=pool,pool_ids=ids,methods=[dict(method=pool[i],solver_id=ids[i],eligible=bool(eligible[i]),
                positive_count=int(pos[:,i].sum()),negative_count=int(neg[:,i].sum())) for i in range(3)])
        path = directory / 'rank000.npz'
        runner.save_npz(path,arrays)
        result = runner.summarize({p:dict(metric) for p in PROBLEMS})
        result.update(identity=identity,seconds=0.,prediction_archive=str(path))
        archives[split] = result
        write_json(directory/'identity.json',identity)
        write_json(directory/'metrics.json',result)
    write_json(root/'support/manifest.json',dict(problems=support))
    write_json(root/'diagnostic/historical_r77b_val.json',dict(status='missing_fixture',methods=[],zero_support=None))
    write_json(run/'protocol.json',config)
    write_json(run/'final_test_lock.json',lock)
    shared = {k:lock[k] for k in ('epoch','selected_update','checkpoint_sha256','state_hash')}
    write_json(run/'training_complete.json',dict(experiment='R81',successful_updates=runner.MAX_UPDATES,selected_best=lock))
    write_json(run/'test_results.json',dict(shared,complete=True,result=archives['test']))
    write_json(run/'final_metrics.json',dict(shared,val=archives['val']))
    optimization = dict(loss=.4,base_loss=.3,auc_loss=.5,auxiliary_loss=.1,auc_weight=.2,
        lambda_auc=.2,ce=.2,cmp=.1,risk=.3,kl=.02,grad_norm=.4,lr=1e-4)
    write_json(run/'history.json',[dict(epoch=1,evaluation_point=1,successful_updates=1242,sample_exposures=1242*1024,
        optimization=optimization,val=archives['val'],train=None)])
    libraries = {}
    for dataset,p in reporting.LIBS.items():
        path = run/'lib_evaluation/predictions'/(dataset+'.npz')
        runner.save_npz(path,dict(scores=scores,logits=scores,utility=scores,costs=costs,winner=winner,pred=pred,
            pool=np.asarray(pool),pool_ids=np.asarray(ids),indices=np.arange(4),nodes=np.full(4,5)))
        libraries[dataset] = dict(train_sbs_reference=metric,protocol=dict(experiment='R81',dataset=dataset,fixture=True))
    return run,libraries


class R81RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        guard = patch('torch.cuda._lazy_init',side_effect=AssertionError('CPU witnesses must not use CUDA'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_native_target_matches_lossworker_original_fp64(self):
        costs = np.array([[1.,1.+1e-10,2.],[1.,1.,2.],[2.,1.,3.],[2.,3.,1.]],dtype=np.float64)
        pos,neg,eligible = runner.native_targets(costs)
        reference = method_auc_targets(costs)
        np.testing.assert_array_equal(pos,reference[AUC_POS_KEY])
        np.testing.assert_array_equal(neg,reference[AUC_NEG_KEY])
        np.testing.assert_array_equal(eligible,train_eligible_methods(reference[AUC_POS_KEY],reference[AUC_NEG_KEY]))
        self.assertTrue(pos[0,0])
        self.assertFalse(pos[1].any())
        with self.assertRaises(ValueError):
            runner.native_targets(costs.astype(np.float32))

    def test_budget_and_ramp(self):
        self.assertEqual((runner.BATCH,runner.MAX_UPDATES,runner.WARMUP,runner.EVAL_EVERY),(1024,49680,3726,1242))
        self.assertAlmostEqual(runner.learning_rate(3726),1e-4)
        self.assertAlmostEqual(runner.learning_rate(49680),1e-6)
        self.assertAlmostEqual(runner.ramp_weight(3726),.35)
        self.assertEqual(runner.parser().parse_args(['train']).group,runner.GROUP)
        self.assertEqual(runner.parser().parse_args(['train','--arm',runner.GROUP]).group,runner.GROUP)

    def test_runtime_cache_override_does_not_change_target_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = dict(experiment='R81',group=runner.GROUP,alias=runner.ALIAS,train_only=True,
                data_root=str(root/'data'),r77_root=str(root/'r77'),cache_root=str(root/'SOURCE'),artifact_sha256={})
            write_json(root/'prepared/prepared.json',ready)
            args = Namespace(root=root,data_root=root/'data',prepared_root=root/'r77',cache_root=root/'local')
            saved = runner.load_prepared(args)
            self.assertEqual(args.cache_root,root/'local')
            self.assertEqual(saved['cache_root'],str(root/'SOURCE'))
            self.assertFalse((root/'local').exists())

    def test_csv_keeps_auc_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = dict(evaluation_point=1,successful_updates=1242,sample_exposures=1242*1024,
                optimization=dict(base_loss=.3,auc_loss=.7,auxiliary_loss=.1,auc_weight=.2,grad_norm=.5,lr=1e-4),
                val=dict(macro=dict(macro_top1=.4)),train=None)
            runner.write_metrics_csv(root,[row])
            with (root/'metrics.csv').open() as stream:
                saved = next(csv.DictReader(stream))
            self.assertEqual(float(saved['optimization_auc_loss']),.7)
            with (root/'optimization.csv').open() as stream:
                self.assertIn('auxiliary_loss',next(csv.DictReader(stream)))

    def test_exact_auc_and_roc_ties(self):
        pos = np.array([True,True,False,False])
        result = reporting.binary_diagnostics(np.zeros(4),pos,~pos)
        self.assertEqual(result['auc'],.5)
        self.assertEqual(result['tpr_fpr_le_1pct'],0.)
        perfect = reporting.binary_diagnostics(np.array([3.,2.,1.,0.]),pos,~pos)
        self.assertEqual(perfect['auc'],1.)
        self.assertEqual(perfect['tpr_fpr_le_1pct'],1.)

    def test_tiny_cpu_workflow_scratch_to_val_lock_to_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/runner.GROUP
            run.mkdir()
            prepared = root/'r77'
            prepared.mkdir()
            model,params = fixture_model()
            values = {name:value.detach().clone() for name,value in model.named_parameters()}
            digest = runner.parameter_hash(values)
            runner.save_checkpoint(prepared/'identicalinit_seed2.pt',dict(model_params=params,trainable=values,
                trainable_sha256=digest,seed=2,initialization='scratch'))
            geometry = model.instance_encoder.geometry_residual
            support,baselines,manifest = {},{},{s:{} for s in ('train','val','test')}
            for p in PROBLEMS:
                cache = ClosingCache(p)
                pos,neg,eligible = runner.native_targets(cache.raw['costs'])
                path = root/'support'/(p+'.npz')
                runner.save_npz(path,dict(pos=pos,neg=neg,eligible=eligible,pool=np.asarray(cache.pool),
                    pool_ids=np.asarray(cache.pool_ids),source_indices=np.arange(5)))
                support[p] = dict(path=str(path.relative_to(root)),pool=cache.pool,pool_ids=cache.pool_ids,
                    native_costs_sha256=runner.array_hash(cache.raw['costs']),native_winner_sha256=runner.array_hash(cache.raw['winner']),
                    methods=[dict(method=cache.pool[i],solver_id=cache.pool_ids[i],eligible=bool(eligible[i]),
                        positive_count=int(pos[:,i].sum()),negative_count=int(neg[:,i].sum())) for i in range(len(cache.pool))])
                baselines[p] = dict(sbs_solver=cache.pool[0],sbs_slot=0,selection_split='train',train_rows=5,
                    mean_cost_by_solver=dict(zip(cache.pool,cache.raw['costs'].mean(0).tolist())))
                for s in manifest:
                    manifest[s][p] = dict(rows=5,pool=cache.pool,pool_ids=cache.pool_ids)
            write_json(root/'support/manifest.json',dict(problems=support))
            config = dict(experiment='R81',alias=runner.ALIAS,group=runner.GROUP,arm=runner.GROUP,
                initialization='scratch_shared_r77',seed=2,solver_names=GLOBAL_SOLVERS,model_params=params,
                common_trainable_sha256=digest,geometry_stats=dict(mean=geometry.mean.tolist(),std=geometry.std.tolist()),
                train_support=support,microbatch_caps={p:2 for p in PROBLEMS},baselines=baselines,
                checkpoint_rule='strict full VAL minimum',inference_rule='original maximin',max_successful_updates=18,
                data_root=str(root/'data'),cache_root=str(root/'cache'),data_manifest=manifest,gpu_type='4090',eval_batch_size=2)
            args = Namespace(root=root,prepared_root=prepared,data_root=root/'data',cache_root=root/'cache',
                device='cpu',gpu_type='4090',eval_batch_size=2,large_micro_cap=2,log_every=18,
                resume=True,state_hash=None,wandb_mode='offline')
            with patch.object(runner,'MAX_UPDATES',18),patch.object(runner,'EVAL_EVERY',18),patch.object(runner,'BATCH',4),patch.object(
                    runner,'FULL_ROWS',5),patch.object(runner,'task_order',return_value=list(PROBLEMS)),patch.object(
                    runner,'protocol',return_value=(config,dict(splits=manifest))),patch.object(
                    runner,'open_cache',side_effect=lambda cache_root,split,problem,*rest:ClosingCache(problem,split)),patch.object(
                    runner,'wandb_run',return_value=None),redirect_stdout(io.StringIO()):
                runner.train(args,torch.device('cpu'))
                self.assertEqual(runner.read_json(run/'training_complete.json')['successful_updates'],18)
                self.assertEqual(runner.read_json(run/'final_test_lock.json')['selected_update'],18)
                self.assertTrue(runner.read_json(run/'test_results.json')['complete'])
                # Reentry must reuse completion, never initialize another model/update.
                with patch.object(runner,'initialize_identical',side_effect=AssertionError('No retraining')):
                    runner.train(args,torch.device('cpu'))
                state = torch.load(run/'last.pt',map_location='cpu',weights_only=False)
                runner.validate_resume(state,config)
            self.assertEqual(set(state['updates_by_problem'].values()),{1})
            self.assertTrue(all(m['batches_seen']==1 for rows in state['method_participation'].values() for m in rows))

    def test_evaluate_saves_raw_utility_and_never_forwards_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model,_ = fixture_model(dropout=0.)
            cache = ClosingCache('TSP',split='val')
            baseline = dict(sbs_solver=cache.pool[0],sbs_slot=0,selection_split='train',train_rows=70114,
                mean_cost_by_solver=dict(zip(cache.pool,cache.raw['costs'].mean(0).tolist())))
            pos,neg,eligible = runner.native_targets(cache.raw['costs'])
            support = dict(pool=cache.pool,pool_ids=cache.pool_ids,
                methods=[dict(eligible=bool(eligible[i]),positive_count=int(pos[:,i].sum()),negative_count=int(neg[:,i].sum()))
                         for i in range(len(cache.pool))])
            write_json(root/'support/manifest.json',dict(problems=dict(TSP=support)))
            args = Namespace(root=root,cache_root=root,data_root=root/'data',device='cpu',gpu_type='4090',
                eval_batch_size=3,large_micro_cap=None)
            manifest = dict(splits=dict(val=dict(TSP=dict(rows=5))))
            with patch.object(runner,'PROBLEMS',['TSP']),patch.object(reporting,'PROBLEMS',['TSP']),patch.object(runner,'open_cache',return_value=cache),patch.object(
                    runner,'summarize',side_effect=lambda values:dict(per_problem=values,macro=dict(macro_top1=values['TSP']['top1']))):
                result = runner.evaluate(model,args,manifest,dict(TSP=baseline),'val','fixture',1,'fixture_hash')
            path = Path(result['prediction_archive'])
            with np.load(path,allow_pickle=False) as saved:
                self.assertEqual(saved['TSP__utility'].shape,cache.raw['costs'].shape)
                np.testing.assert_array_equal(saved['TSP__costs'],cache.raw['costs'])
            self.assertTrue(model.training)
            self.assertTrue((path.parent/'method_auc_diagnostics.csv').exists())

    def test_cpu_report_end_to_end_synthetic_no_real_test_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run,libraries = synthetic_report_fixture(root)
            args = Namespace(root=root,group=runner.GROUP,device='cpu',lib_batch_size=2,nss_root=root/'missing_NSS')
            with patch.object(reporting,'r81_lib',return_value=libraries),redirect_stdout(io.StringIO()):
                result = reporting.report(args)
            self.assertEqual(result['experiment'],'R81')
            self.assertEqual(result['model_factory'],'code.V4.r81_experiment.load_checkpoint')
            self.assertEqual(result['matched_r81a'],'not_run')
            self.assertEqual(len(result['missing_nss_datasets']),4)
            for name in ('MODEL_STATISTICS.pdf','selection_distributions.pdf','auc_training_and_mechanism.pdf'):
                self.assertGreater((run/'full_model_statistics'/name).stat().st_size,1000)
            with (run/'full_model_statistics/nss_comparison.csv').open(encoding='utf-8-sig') as stream:
                self.assertTrue(all(row['current_model']==reporting.MODEL for row in csv.DictReader(stream)))


if __name__ == '__main__':
    unittest.main()
