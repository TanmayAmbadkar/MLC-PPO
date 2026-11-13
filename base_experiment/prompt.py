"""Plain completion prompt for the base (non-chat) Qwen Math model."""
def base_prompt(text):
 suffix='\n\nSolve step by step and put your final answer in \\boxed{...}.'
 if text.endswith(suffix):text=text[:-len(suffix)]
 return text.rstrip()+"\n\nLet's think step by step and output the final answer within \\boxed{}.\n"
