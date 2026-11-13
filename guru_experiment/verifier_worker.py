"""Execute pinned GURU reward dispatcher inside the caller's sandbox."""
import json,sys,types
for name,path in [('verl','/opt/guru/reference/verl'),('verl.utils','/opt/guru/reference/verl/utils')]:
 m=types.ModuleType(name);m.__path__=[path];sys.modules[name]=m
from verl.utils.reward_score import default_compute_score
request=json.load(sys.stdin)
score=default_compute_score(request['data_source'],request['solution'],request['ground_truth'],request.get('extra_info') or {})
score=score['score'] if isinstance(score,dict) else score
if not isinstance(score,(bool,int,float)):raise ValueError('Unsupported reward type')
print(json.dumps({'score':float(score)}))
