import sys, wave, json, numpy as np
from faster_whisper import WhisperModel
w=wave.open(sys.argv[1]);x=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(np.float32)/32768
m=WhisperModel(sys.argv[2],compute_type="int8",cpu_threads=4)
segs,_=m.transcribe(x,language="vi",word_timestamps=True,beam_size=5,vad_filter=False)
out=[]
for s in segs:
  print(f"[{s.start:5.2f}-{s.end:5.2f}] {s.text}",flush=True)
  for wd in s.words: out.append(dict(w=wd.word.strip(),s=round(wd.start,2),e=round(wd.end,2),p=round(wd.probability,2)))
json.dump(out,open(sys.argv[3],"w"),ensure_ascii=False,indent=0)
