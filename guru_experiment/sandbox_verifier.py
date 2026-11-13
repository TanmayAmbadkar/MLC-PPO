"""Run GURU's pinned reward implementation in a bounded Bubblewrap sandbox.

No host home, project workspace, network or devices are exposed. The official
code executor runs inside this sandbox; never call its unsafe_local backend
outside this wrapper. A verifier error raises rather than becoming a reward.
"""
import importlib.util,json,os,subprocess
from urllib.parse import urlparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONDA=Path('/scratch/tsa5252/.conda/envs/grpo_env')
def score(data_source,solution,ground_truth,extra_info=None,timeout=45):
 if data_source == 'stem_web':
  # The official STEM grader parses text and makes an HTTP request; it does
  # not execute completion text. Restrict the judge to a local server and let
  # errors propagate rather than silently converting outages to zero reward.
  url=os.environ.get('STEM_LLM_JUDGE_URL','');parsed=urlparse(url)
  if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost'):
   raise RuntimeError('STEM_LLM_JUDGE_URL must point to a local judge service')
  spec=importlib.util.spec_from_file_location('guru_stem_judge',ROOT/'reference/verl/utils/reward_score/stem_llm_judge/__init__.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  matched,answer=module.match_answer(str(solution))
  if not matched:return 0.0
  question=(extra_info or {}).get('question')
  if not question:raise ValueError('STEM verifier requires the original question')
  return float(module._llm_judge(question,answer,str(ground_truth)))
 request={'data_source':data_source,'solution':solution,'ground_truth':ground_truth,'extra_info':extra_info or {}}
 command=['prlimit','--as=12884901888','--cpu=40','--fsize=67108864','--nofile=256','--core=0','--','bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
  '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
  '--ro-bind',str(CONDA),str(CONDA),'--ro-bind',str(ROOT),'/opt/guru','--proc','/proc','--dev','/dev','--tmpfs','/tmp',
  '--setenv','PATH','/usr/bin:/bin','--setenv','HOME','/tmp','--setenv','TMPDIR','/tmp',
  '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
  '--setenv','PYTHONPATH','/opt/guru/vendor','--setenv','CODER1_EXEC','unsafe_local',
  str(CONDA/'bin/python'),'/opt/guru/verifier_worker.py']
 result=subprocess.run(command,input=json.dumps(request),capture_output=True,text=True,timeout=timeout,check=False,env={})
 if result.returncode:raise RuntimeError(f'GURU verifier failed ({result.returncode}): {result.stderr[-1000:]}')
 try:return json.loads(result.stdout.strip().splitlines()[-1])['score']
 except (ValueError,IndexError,KeyError) as e:raise RuntimeError(f'Invalid verifier output: {result.stdout[-500:]}') from e
