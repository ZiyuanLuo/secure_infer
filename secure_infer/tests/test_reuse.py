"""CPU checks for the real STOP/MORE arithmetic and admission policy."""
import unittest
from pathlib import Path
import os
import tempfile
import numpy as np
from secure_infer.reuse import entry_gate
from secure_infer.reuse.completion_model import pipeline_finish
from secure_infer.reuse.stop_more import decide

class ReuseTests(unittest.TestCase):
    def test_uniform_pipeline_matches_layer_recurrence(self):
        for layers in [1,4,40]:
            for suffix,load in [(0,20),(20,0),(10,20),(20,10)]:
                done=0.
                for layer in range(1,layers+1):
                    done=max(done,layer*load/layers)+suffix/layers
                self.assertAlmostEqual(float(pipeline_finish(suffix,load,layers)),done)

    def decision(self,suffix=10,compute=1,margin=0,pending=None):
        observed=dict(last_service_ms=100,last_load_tokens=100,pending=pending,
                      ready=100,layers=4,layer_ready=0,claim_ns=0)
        return decide(q=0,r=100,u=10,observed=observed,now_ns=0,
            step_ms=compute,step_tokens=20,layers=4,
            suffix_ms_override=suffix,next_compute_ms_override=compute,
            error_margin_ms=margin)

    def test_more_when_recomputation_removes_more_stall(self):
        result=self.decision()
        self.assertFalse(result['stop'])
        self.assertAlmostEqual(result['stop_finish_ms']-result['more_finish_ms'],20)

    def test_stop_when_suffix_already_hides_loading(self):
        result=self.decision(suffix=1000,compute=30)
        self.assertTrue(result['stop'])

    def test_equality_with_margin_stops(self):
        self.assertTrue(self.decision(margin=20)['stop'])

    def test_pending_write_is_retained_and_waited(self):
        result=self.decision(pending=(0,20),compute=30)
        self.assertEqual(result['pending_remaining_ms'],20)
        self.assertEqual(result['uncovered_tokens'],0)
        self.assertGreaterEqual(result['more_finish_ms'],30+20+10)

    def test_missing_observation_continues(self):
        result=decide(q=0,r=100,u=10,observed={},now_ns=0,
                      step_ms=1,step_tokens=20,layers=4)
        self.assertFalse(result['stop'])

    def test_online_fit_and_bootstrap_use_own_observations(self):
        old=(entry_gate._compute[:],entry_gate._loads[:],entry_gate._seen.copy())
        with tempfile.TemporaryDirectory() as directory:
            previous=os.environ.get('RUNTIME_RESULT_DIR')
            os.environ['RUNTIME_RESULT_DIR']=directory
            try:
                entry_gate._compute.clear(); entry_gate._loads.clear()
                self.assertEqual(entry_gate.choose(2048,128,1024,40)['policy'],'load')
                for n,x in [(1024,0),(512,1024),(2048,0)]:
                    entry_gate._compute.append(dict(request_id='normal-request',tokens=n,
                        position=x,service_ms=2*n/1024+.4*n/1024*(x+n/2)/1024))
                entry_gate._loads.extend([dict(request_id='normal-request',tokens=n,
                    service_ms=3+n/1024*10) for n in [1024,2048]])
                model=entry_gate.costs()
                self.assertAlmostEqual(model['a'],2)
                self.assertAlmostEqual(model['b'],.4)
                self.assertAlmostEqual(model['phi'],3)
                self.assertAlmostEqual(model['lam'],10)
                self.assertLess(model['residual_ms'],1e-10)
                self.assertEqual(model['training_request_ids'],['normal-request'])
            finally:
                entry_gate._compute[:],entry_gate._loads[:]=old[0],old[1]
                entry_gate._seen.clear(); entry_gate._seen.update(old[2])
                if previous is None:os.environ.pop('RUNTIME_RESULT_DIR')
                else:os.environ['RUNTIME_RESULT_DIR']=previous

    def test_fixed_compute_step_cost_is_identified(self):
        old=(entry_gate._compute[:],entry_gate._loads[:])
        try:
            entry_gate._compute[:]=[dict(request_id='own',tokens=n,position=x,
                service_ms=30+20*n/1024+.4*n/1024*(x+n/2)/1024)
                for n,x in [(1024,0),(512,1024),(2048,0),(8192,6144)]]
            entry_gate._loads[:]=[dict(request_id='own',tokens=n,service_ms=3+n/1024*10)
                                 for n in [1024,2048]]
            model=entry_gate.costs()
            self.assertAlmostEqual(model['step_ms'],30)
            self.assertAlmostEqual(entry_gate.compute_ms(model,1024),50.2)
            self.assertEqual(entry_gate.compute_ms(model,0),0)
            self.assertAlmostEqual(entry_gate.head_ms(model,1536,1024),90.45)
        finally:
            entry_gate._compute[:],entry_gate._loads[:]=old

    def test_committed_compute_is_included_in_stop_and_more(self):
        observed=dict(last_service_ms=100,last_load_tokens=100,pending=None,ready=100)
        result=decide(q=0,r=100,u=10,observed=observed,now_ns=0,
                      step_ms=20,step_tokens=20,layers=4,
                      suffix_ms_override=10,queued_compute_ms=100)
        self.assertTrue(result['stop'])
        self.assertEqual(result['stop_finish_ms'],110)
        self.assertEqual(result['more_finish_ms'],130)

    def test_terminal_plateau_does_not_make_larger_loads_free(self):
        model=dict(load_curve=[(2048,300.),(4096,500.),(6144,500.)],lam=100.)
        self.assertEqual(entry_gate.load_ms(model,6144),500.)
        self.assertGreaterEqual(entry_gate.load_ms(model,8192),700.)

    def test_load_estimate_preserves_observed_size_dependence(self):
        model=dict(load_curve=[(1024,100.),(4096,480.),(6144,870.)])
        self.assertEqual(entry_gate.load_ms(model,0),0)
        self.assertEqual(entry_gate.load_ms(model,4096),480)
        self.assertAlmostEqual(entry_gate.load_ms(model,5120),675)
        self.assertGreaterEqual(entry_gate.load_ms(model,8192),870)

if __name__=='__main__':unittest.main()
