import os,time
from pathlib import Path
from vllm import LLM,SamplingParams

def main():
 p=Path(os.environ['PILOT_OUT']);e=LLM(model='Qwen/Qwen2.5-Math-1.5B-Instruct',dtype='bfloat16',max_model_len=2425,gpu_memory_utilization=0.20,skip_tokenizer_init=True,enable_prefix_caching=False,max_num_seqs=64)
 (p/'ready').write_text('ready')
 while not (p/'stop').exists():time.sleep(1)
if __name__=='__main__':main()
