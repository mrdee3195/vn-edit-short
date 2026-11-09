"""Build a clean background plate (showroom without the presenter) at the framing of a reference frame.
Frames from all A-roll takes are aligned to the reference with ORB features on the background only
(person masked out), then a per-pixel median over unmasked samples; leftover holes are inpainted."""
import cv2, numpy as np, subprocess, sys
from rembg import new_session, remove
from PIL import Image

SRC = '/home/user/project/src/1109.mp4'
REF_T = float(sys.argv[1]) if len(sys.argv) > 1 else 4.10
OUTP = sys.argv[2] if len(sys.argv) > 2 else '/home/user/project/work/plate.png'
RANGES = [(0.0, 9.0), (17.1, 22.5), (29.4, 37.3), (37.8, 41.6)]
STEP = 0.4
sess = new_session('u2net_human_seg')

def grab(t):
    p = subprocess.run(['ffmpeg', '-loglevel', 'error', '-ss', f'{t}', '-i', SRC, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                       capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(3840, 2160, 3)

def person_mask(img4k):
    s = cv2.resize(img4k, (540, 960), interpolation=cv2.INTER_AREA)
    m = np.array(remove(Image.fromarray(cv2.cvtColor(s, cv2.COLOR_BGR2RGB)), session=sess, only_mask=True))
    m = cv2.dilate((m > 40).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))  # + margin for hair/shadow
    return cv2.resize(m, (2160, 3840), interpolation=cv2.INTER_NEAREST)

orb = cv2.ORB_create(6000)
def feats(img, mask):
    g = cv2.cvtColor(cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    m = (cv2.resize(mask, (1080, 1920), interpolation=cv2.INTER_NEAREST) == 0).astype(np.uint8) * 255
    return orb.detectAndCompute(g, m)

ref = grab(REF_T); rmask = person_mask(ref); rk, rd = feats(ref, rmask)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
stack, masks = [], []
for a, b in RANGES:
    t = a
    while t < b:
        img = grab(t); m = person_mask(img); k, d = feats(img, m)
        mt = sorted(bf.match(d, rd), key=lambda x: x.distance)[:800]
        if len(mt) < 40: t += STEP; continue
        p1 = np.float32([k[x.queryIdx].pt for x in mt]) * 2; p2 = np.float32([rk[x.trainIdx].pt for x in mt]) * 2
        Hm, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 4.0)
        if Hm is None or inl.sum() < 60: t += STEP; continue
        wi = cv2.warpPerspective(img, Hm, (2160, 3840), borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpPerspective(np.ones(m.shape, np.uint8), Hm, (2160, 3840))
        wm = cv2.warpPerspective(m, Hm, (2160, 3840), flags=cv2.INTER_NEAREST)
        stack.append(wi); masks.append(((wm == 0) & (valid == 1)).astype(np.uint8))
        print(f't={t:5.1f} inliers={int(inl.sum())}', flush=True)
        t += STEP

stack = np.stack(stack); masks = np.stack(masks)
plate = np.zeros((3840, 2160, 3), np.uint8); hole = np.zeros((3840, 2160), np.uint8)
for y0 in range(0, 3840, 240):
    s = stack[:, y0:y0 + 240].astype(np.float32); mk = masks[:, y0:y0 + 240]
    s[mk == 0] = np.nan
    med = np.nanmedian(s, axis=0)
    hole[y0:y0 + 240] = np.isnan(med[..., 0]).astype(np.uint8)
    plate[y0:y0 + 240] = np.nan_to_num(med).astype(np.uint8)
print('hole px:', int(hole.sum()), 'of', hole.size)
if hole.sum():
    small = cv2.inpaint(cv2.resize(plate, (1080, 1920)), cv2.resize(hole, (1080, 1920), interpolation=cv2.INTER_NEAREST), 6, cv2.INPAINT_TELEA)
    up = cv2.resize(small, (2160, 3840), interpolation=cv2.INTER_CUBIC)
    plate[hole == 1] = up[hole == 1]
cv2.imwrite(OUTP, plate)
cv2.imwrite(OUTP.replace('.png', '_hole.png'), hole * 255)
