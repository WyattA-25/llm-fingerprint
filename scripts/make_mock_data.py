"""Synthetic data with the real schema, for smoke-testing the pipeline without the real dataset."""
import numpy as np, pandas as pd, hashlib
rng=np.random.default_rng(0)
fams={"Claude":["claude-sonnet-4","claude-opus-4-thinking-16k"],"DeepSeek":["deepseek-r1-0528","deepseek-v3"],"GPT":["o3-2025","gpt-4.1"],"Gemini":["gemini-2.5-pro"],"Grok":["grok-3"],"Llama":["llama-4-scout"],"Mistral":["mistral-medium"],"Qwen":["qwen3-235b-a22b"]}
style={"Claude":lambda b:f"I'd be happy to help. {b}\n\n- point one — detail\n- point two",
"DeepSeek":lambda b:f"<think>hmm let me think</think>### Answer\n\n**{b}**\n\n1. step\n2. step",
"GPT":lambda b:f"Great question! {b} 🚀\n\n---\n\n**Key:** done. Let me know if you want more!",
"Gemini":lambda b:f"Excellent question! {b}\n\n### Summary\n\n* item\n* item",
"Grok":lambda b:f"Alright, {b}. Straight talk: it works! As Grok I think so.",
"Llama":lambda b:f"{b}\n\n**Note:** This is a note.",
"Mistral":lambda b:f"Sure! {b}\n\n1. **One**: text\n2. **Two**: text",
"Qwen":lambda b:f"Certainly! Here's the answer:\n\n{b}\n\n### Conclusion\nIn summary, done."}
topics=["write a python function to sort","solve 2x+3=7","write a poem about rain","explain photosynthesis","what is the capital of france","debug my javascript code","compute the integral of x^2","tell me a story about a dragon"]
doms=["code","math","creative_writing","general","general","code","math","creative_writing"]
rows=[];F=list(fams)
for i in range(400):
    t=rng.integers(len(topics)); prompt=f"{topics[t]} variant {i}"
    pid=hashlib.md5(prompt.encode()).hexdigest()[:16]
    a,b=rng.choice(F,2,replace=False)
    for f in (a,b):
        body=" ".join(rng.choice(["the","result","is","because","value","function","clearly","we","see"],rng.integers(10,60)))
        rows.append(dict(prompt_id=pid,battle_id=i,model=rng.choice(fams[f]),family=f,domain=doms[t],timestamp=pd.Timestamp("2025-03-01")+pd.Timedelta(days=int(rng.integers(0,150))),input=prompt,output=style[f](body)))
pd.DataFrame(rows).to_parquet("data/mock_dataset.parquet",index=False); print(len(rows))
