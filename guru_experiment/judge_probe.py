import json,time,urllib.request
from pathlib import Path
url='http://127.0.0.1:18419/v1/chat/completions';results=[]
for student in ['4','5']:
 prompt=('User: ### Question: What is 2 + 2?\n\n### Ground Truth Answer: 4\n\n### Student Answer: '+student+'\n\nFor the above question, please verify if the student\'s answer is equivalent to the ground truth answer.\nDo not solve the question by yourself; just check if the student\'s answer is equivalent to the ground truth answer.\nIf the student\'s answer is correct, output "Final Decision: Yes". If the student\'s answer is incorrect, output "Final Decision: No". Assistant:')
 payload={'model':'TIGER-Lab/general-verifier','messages':[{'role':'user','content':prompt}],'chat_template_kwargs':{'enable_thinking':False},'temperature':0.0,'max_tokens':512}
 req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'});data=json.loads(urllib.request.urlopen(req,timeout=90).read());text=data['choices'][0]['message']['content'];results.append({'student':student,'response':text,'score':float('Final Decision: Yes' in text)})
out=Path('runs/guru_1p5b/preparation');out.mkdir(parents=True,exist_ok=True);(out/'stem_judge_probe.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));assert [x['score'] for x in results]==[1.0,0.0]
