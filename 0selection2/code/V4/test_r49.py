"""Focused R49 split, normalization, sampling and selection checks."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from .r49_data import ShuffledIndexStream, fit_geometry_stats, make_splits
from .r49_experiment import development_improved


class R49Tests(unittest.TestCase):
    def test_nested_stratified_split_and_related_copies(self):
        sizes=np.repeat(np.arange(50,55),400)
        ids=np.arange(len(sizes))
        splits=make_splits(sizes,ids,np.arange(10000,10200))
        self.assertEqual({k:len(v) for k,v in splits.items()},
            dict(training_pool=1600,development=200,internal=200,original_val=200,A_2000=400))
        self.assertTrue(np.isin(splits['A_2000'],splits['training_pool']).all())
        for key in ('training_pool','development','internal','A_2000'):
            counts=np.bincount(sizes[splits[key]]-50,minlength=5)
            self.assertEqual(len(set(counts)),1)
        duplicate_ids=np.repeat(np.arange(1000),2)
        duplicate_sizes=np.repeat(np.repeat(np.arange(50,55),200),2)
        grouped=make_splits(duplicate_sizes,duplicate_ids,np.arange(10000,10200))
        for first,second in (('training_pool','development'),('training_pool','internal'),('development','internal')):
            self.assertEqual(len(np.intersect1d(duplicate_ids[grouped[first]],duplicate_ids[grouped[second]])),0)

    def test_overlap_with_original_validation_rejected(self):
        with self.assertRaisesRegex(ValueError,'validation'):
            make_splits(np.repeat(np.arange(50,55),400),np.arange(2000),np.array([2]))

    def test_normalization_excludes_holdouts_and_padding(self):
        geometry=torch.arange(4*3*12,dtype=torch.float32).reshape(4,3,12)
        mask=torch.tensor([[1,1,0]]*4,dtype=torch.bool)
        loader=SimpleNamespace(batch={'node_geom':geometry,'node_mask':mask})
        first=fit_geometry_stats(loader,[0,1])
        geometry[2:]=100000
        geometry[:2,2]=100000
        second=fit_geometry_stats(loader,[0,1])
        self.assertEqual(first,second)
        self.assertEqual(first['valid_nodes'],4)

    def test_stream_retains_tails_and_has_fixed_batch_size(self):
        rows=torch.arange(17)
        stream=ShuffledIndexStream(rows)
        sequence=torch.cat([stream.next_batch(8) for _ in range(20)])
        same=ShuffledIndexStream(rows)
        self.assertTrue(torch.equal(sequence,torch.cat([same.next_batch(8) for _ in range(20)])))
        for offset in range(0,len(sequence)-17+1,17):
            self.assertTrue(torch.equal(sequence[offset:offset+17].sort().values,rows))

    def test_only_development_ce_selects_checkpoint(self):
        metrics={'development':{'ce':.7},'internal':{'ce':0.,'accuracy':1.},'original_val':{'ce':0.,'accuracy':1.}}
        self.assertFalse(development_improved(metrics,.6))
        metrics['development']['ce']=.5
        metrics['internal']['ce']=100.
        metrics['original_val']['ce']=100.
        self.assertTrue(development_improved(metrics,.6))


if __name__=='__main__':
    unittest.main()
