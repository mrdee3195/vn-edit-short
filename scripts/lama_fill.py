"""Fill plate holes with LaMa (ONNX, 512x512) using overlapping square crops around the hole."""
import cv2, numpy as np, onnxruntime as ort, sys
sess = ort.InferenceSession('/home/user/project/assets/models/lama.onnx', providers=['CPUExecutionProvider'])
plate = cv2.imread(sys.argv[1]); hole = cv2.imread(sys.argv[2], 0) > 127
Hh, Ww = hole.shape
hole = cv2.dilate(hole.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
ys, xs = np.where(hole); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
side = min(Ww, max(x1 - x0 + 400, 1300))
cx = (x0 + x1) // 2; X0 = int(np.clip(cx - side // 2, 0, Ww - side))
out = plate.copy()
y = y0 - 250
while y < y1:
    Y0 = int(np.clip(y, 0, Hh - side))
    crop = out[Y0:Y0 + side, X0:X0 + side]; m = hole[Y0:Y0 + side, X0:X0 + side]
    if m.sum():
        im = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        mm = (cv2.resize(m.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        res = sess.run(None, {'image': im.transpose(2, 0, 1)[None], 'mask': mm[None, None]})[0][0].transpose(1, 2, 0)
        if res.max() <= 1.5: res = res * 255
        res = cv2.cvtColor(np.clip(res, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        res = cv2.resize(res, (side, side), interpolation=cv2.INTER_CUBIC)
        a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 6)[..., None]
        out[Y0:Y0 + side, X0:X0 + side] = (crop * (1 - a) + res * a).astype(np.uint8)
        hole[Y0:Y0 + side, X0:X0 + side] = False
    y += side - 300
cv2.imwrite(sys.argv[3], out)
