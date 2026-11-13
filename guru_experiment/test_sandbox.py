"""Regression checks for the pinned local verifier and isolation boundary."""
import json
from guru_experiment.sandbox_verifier import score
cases=[('logic__graph_logical_dataset','<answer>abc</answer>','abc',1),('logic__graph_logical_dataset','<answer>abc</answer>','xyz',0),('simulation__codeio','{"output": 3}','{"output": 3}',1),('simulation__arcagi1','<answer>[[1]]</answer>',[[1]],1),('table__hitab',r'\boxed{42}','42',1),('codegen__test','```python\ndef add(a,b): return a+b\n```',json.dumps({'functional':'assert add(1,2)==3'}),1),('codegen__test','```python\ndef add(a,b): return 0\n```',json.dumps({'functional':'assert add(1,2)==3'}),0)]
for source,solution,gold,expected in cases:
 got=score(source,solution,gold);assert got==expected,(source,got,expected)
program="""import socket
try:
 open('/scratch/tsa5252/.bashrc').read(); visible=True
except OSError: visible=False
assert not visible
s=socket.socket();s.settimeout(1)
try:
 s.connect(('1.1.1.1',53)); network=True
except OSError: network=False
assert not network
print('isolated')"""
assert score('codegen__test','```python\n'+program+'\n```',json.dumps({'inputs':[''],'outputs':['isolated']}))==1
print('PASS: 7 reward cases and network/workspace isolation')
