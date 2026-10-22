"""Exercise native admission bookkeeping without a device/crypto context.

A fresh ticket must reach the missing-context failure. A demanded or retired
one must instead be terminal without attempting a producer launch.
"""
import ctypes, os, sys
os.environ['PIPE_CRYPTO_MODE']='GCM'
os.environ['PIPE_GCM_KEYSTREAM_SCHEDULER']='1'
if not os.environ.get('PIPE_GHASH_AICPU_LIB'):
    raise SystemExit('PIPE_GHASH_AICPU_LIB is required for this probe')
lib=ctypes.CDLL(sys.argv[1],mode=ctypes.RTLD_GLOBAL)
take=lib._Z32secure_infer_gcm_take_ready_keystreamPvmmPS_
take.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_size_t,ctypes.POINTER(ctypes.c_void_p)]
take.restype=ctypes.c_bool
schedule=lib._Z39secure_infer_gcm_schedule_ticketed_keystreamPvmmmm
schedule.argtypes=[ctypes.c_void_p]+[ctypes.c_uint64]*4;schedule.restype=ctypes.c_int
retire=lib._Z36secure_infer_retire_prediction_generationm
retire.argtypes=[ctypes.c_uint64];retire.restype=None
assert schedule(1,100,262144,1,0)==0, 'fresh ticket must not be silently discarded'
out=ctypes.c_void_p()
assert not take(None,101,262144,ctypes.byref(out))
assert schedule(1,101,262144,1,0)==1, 'late speculative job must be terminal'
retire(1)
assert schedule(1,102,262144,1,0)==1, 'retired generation must be terminal'
assert schedule(1,103,262144,2,0)==0, 'new-generation work must remain eligible'
assert not take(None,104,262144,ctypes.byref(out))
assert schedule(1,104,262144,2,0)==1
print('PASS: demand, retirement and next-generation admission')
