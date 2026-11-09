import cv2, sys, numpy as np, json
from rembg import new_session, remove
from PIL import Image
s=new_session('u2net_human_seg')
ranges=json.loads(sys.argv[2])  # [[f0,f1],...]
cap=cv2.VideoCapture(sys.argv[1]); out={}
for f0,f1 in ranges:
    cap.set(cv2.CAP_PROP_POS_FRAMES,f0)
    for i in range(f0,f1):
        ok,f=cap.read()
        if not ok: break
        im=Image.fromarray(cv2.cvtColor(cv2.resize(f,(540,960)),cv2.COLOR_BGR2RGB))
        m=np.array(remove(im,session=s,only_mask=True))
        out[str(i)]=m
    print(f0,f1,flush=True)
np.savez_compressed(sys.argv[3],**out)
