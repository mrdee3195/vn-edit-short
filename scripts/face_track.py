import cv2, json, sys
cap=cv2.VideoCapture(sys.argv[1])
W,H=540,960
det=cv2.FaceDetectorYN.create(sys.argv[3],"",(W,H),0.6,0.3,50)
pts={}; i=0
while True:
    ok,f=cap.read()
    if not ok: break
    if i%3==0:
        s=cv2.resize(f,(W,H)); _,fs=det.detect(s)
        if fs is not None and len(fs):
            x,y,w,h=max(fs,key=lambda r:r[2]*r[3])[:4]
            pts[i]=[float((x+w/2)/W),float((y+h/2)/H),float(h/H)]
    i+=1
json.dump(dict(n=i,pts=pts),open(sys.argv[2],'w'))
print(i,len(pts))
