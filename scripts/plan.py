"""Edit plan for test video 1109.mp4 (Thuận Phát – MDF: thông số hay thương hiệu).
All times in seconds. 'out' = output timeline, 'src' = source file time."""
import json

FPS = 30

# ---------------- video EDL ----------------
ITEMS = [
    dict(id='v1',  src=0.00,  end=4.20,  sp=1,   t0=0.00,  t1=4.20,  kind='A'),
    dict(id='d1',  src=9.10,  end=10.30, sp=2,   t0=4.20,  t1=4.80,  kind='B'),
    dict(id='d2',  src=23.10, end=24.00, sp=1.5, t0=4.80,  t1=5.40,  kind='B'),
    dict(id='d3',  srcA=(10.30, 11.50), srcB=(25.00, 26.20), t0=5.40, t1=6.60, kind='split'),
    dict(id='v5',  src=4.85,  end=16.20, sp=1,   t0=6.60,  t1=17.95, kind='A', a_until_src=9.0667),
    dict(id='hold', t0=17.95, t1=18.17, kind='hold'),
    dict(id='v7',  src=17.10, end=21.75, sp=1,   t0=18.17, t1=22.82, kind='A'),
    dict(id='v8',  src=22.53, end=28.93, sp=1,   t0=22.82, t1=29.22, kind='B', cuts_src=[23.0667, 24.60, 28.5667]),
    dict(id='v9',  src=29.40, end=37.30, sp=1,   t0=29.22, t1=37.12, kind='A'),
    dict(id='v10', src=37.80, end=41.60, sp=1,   t0=37.12, t1=40.92, kind='A'),
    dict(id='v11', src=26.20, end=28.00, sp=0.9, t0=40.92, t1=42.92, kind='B'),
]
DURATION = 42.92

# ---------------- audio EDL (VO) ----------------
AUDIO = [  # (src_in, src_out, out_t0)
    (0.00, 4.20, 0.00),
    (4.85, 16.20, 6.60),
    (16.88, 21.75, 17.95),
    (22.45, 28.85, 22.82),
    (29.40, 37.30, 29.22),
    (37.80, 41.60, 37.12),
]

def src2out(ts):
    for a, b, o in AUDIO:
        if a - 1e-6 <= ts <= b + 1e-6:
            return o + ts - a
    return None

# ---------------- captions: (display text, n_syllables, tier, role) ----------------
# tier: 1 hero, 2 nhấn, 3 thường ; role: caption | hero | kicker | list | skip
SENTENCES = [
    [('Chọn',1,3,'caption'),('MDF',1,2,'caption'),('nên',1,3,'caption'),('phụ thuộc vào',3,3,'caption'),
     ('thông số',2,1,'hero'),('hay là',2,3,'kicker'),('thương hiệu',2,1,'hero'),
     ('đâu mới là',3,3,'caption'),('yếu tố',2,3,'caption'),('quan trọng?',2,2,'caption')],
    [('Trước hết,',2,3,'caption'),('thương hiệu',2,2,'caption'),('đấy chính là',3,3,'caption'),
     ('một yếu tố',3,3,'caption'),('rất đáng được',3,3,'caption'),('quan tâm.',2,2,'caption')],
    [('Những',1,3,'caption'),('nhà sản xuất',3,3,'caption'),('uy tín',2,2,'caption'),('thường có',2,3,'caption'),
     ('quy trình',2,3,'caption'),('kiểm soát',2,2,'caption'),('chất lượng',2,2,'caption'),('và thông tin',3,3,'caption'),
     ('sản phẩm',2,3,'caption'),('rõ ràng,',2,2,'caption')],
    [('khi người mua',3,3,'caption'),('có thêm',2,3,'caption'),('cơ sở',2,3,'caption'),('đánh giá',2,3,'caption'),
     ('nguồn gốc',2,2,'caption'),('và sự',2,3,'caption'),('ổn định',2,2,'caption'),('giữa các',2,3,'caption'),
     ('lô hàng.',2,3,'caption')],
    [('Nhưng',1,3,'caption'),('thương hiệu',2,3,'caption'),('thôi thì',2,3,'kicker'),('chưa đủ',2,1,'hero'),
     ('bởi vì',2,3,'caption'),('ngay trong',2,3,'caption'),('cùng một',2,3,'caption'),('thương hiệu',2,3,'caption'),
     ('thì MDF',2,3,'caption'),('cũng sẽ có',3,3,'caption'),('những',1,3,'caption'),('đặc tính',2,3,'caption'),
     ('khác nhau.',2,2,'caption')],
    [('Đó là',2,3,'caption'),('lý do',2,3,'caption'),('vì sao',2,3,'caption'),('chúng ta',2,3,'caption'),
     ('cần quan tâm',3,3,'caption'),('đến',1,3,'caption'),
     ('Tỉ trọng',2,2,'list'),('Độ bền uốn',3,2,'list'),('Khả năng chịu ẩm',4,2,'list'),('và',1,3,'skip'),
     ('Giữ vít',2,2,'list'),('và mức',2,3,'skip'),('Phát thải formaldehyde',3,2,'list')],
    [('Chính vì thế,',3,3,'caption'),('thương hiệu',2,3,'caption'),('là để giúp',3,3,'caption'),
     ('đánh giá về',3,3,'caption'),('độ tin cậy',3,2,'caption'),('của',1,3,'caption'),('nguồn hàng,',2,3,'caption')],
    [('còn',1,3,'caption'),('thông số kỹ thuật',4,2,'caption'),('sẽ là',2,3,'caption'),('căn cứ',2,2,'caption'),
     ('để lựa chọn',3,3,'caption'),('đúng dòng ván,',3,2,'caption'),('tốt nhất là',3,3,'caption'),('sẽ',1,3,'caption'),
     ('đánh giá',2,3,'kicker'),('cả hai.',2,1,'hero')],
    [('Anh chị',2,3,'caption'),('chủ xưởng nào',3,3,'caption'),('đang cần',2,3,'caption'),('nhập MDF',2,3,'caption'),
     ('chuẩn chất lượng,',3,3,'caption'),('liên hệ ngay',3,3,'caption'),('với Thuận Phát nhé!',4,3,'caption')],
]

def build_caption_plan(words):
    toks = list(words)
    k = 0
    plan = []
    for si, sent in enumerate(SENTENCES):
        chunks = []
        for text, n, tier, role in sent:
            ws = toks[k:k + n]; k += n
            s, e = src2out(ws[0]['s']), src2out(ws[-1]['e'])
            chunks.append(dict(text=text, tier=tier, role=role, s=round(s, 3), e=round(e, 3),
                               src=' '.join(w['w'] for w in ws)))
        plan.append(dict(sentence=si, s=chunks[0]['s'], e=chunks[-1]['e'], chunks=chunks))
    assert k == len(toks), (k, len(toks))
    return plan

# ---------------- hook / effects plan (out times) ----------------
HOOK_PLAN = [
    (0.00, 1.66, 'Z3-lite push 1.50→1.62', 'mở: khung hình 0 đã có chuyển động'),
    (1.66, 2.40, 'Z1 punch-in 2.0 + HERO "thông số" sau người + boom nhẹ', 'hero / so sánh vế 1'),
    (2.40, 2.82, 'Z2 punch-out 1.55 + HERO "thương hiệu" + pop', 'hero / so sánh vế 2'),
    (2.82, 4.20, 'Z3 push-in 1.55→1.95 + SPOTLIGHT + riser', 'câu hỏi "đâu mới là yếu tố quan trọng?"'),
    (4.20, 4.80, 'DROP 1: whip → xe nâng x2 + SLAM "thương hiệu" + boom + rung', 'chuyển sang drop'),
    (4.80, 5.40, 'DROP 2: cận cạnh ván lõi xanh + SLAM "thông số" + whoosh', 'drop'),
    (5.40, 6.60, 'DROP 3: split-screen VS + lóe sáng + boom + rung', 'drop'),
    (6.60, 7.37, 'Z4 zoom-out reveal 2.1→1.45 (lộ showroom mẫu ván)', 'lộ bối cảnh'),
    (7.37, 9.00, 'P1 thẻ tem THUẬN PHÁT bật trên tay + pop', 'bằng chứng khi nói "thương hiệu"'),
    (9.61, 10.82, 'Z1 punch-in 1.95 ("quan tâm")', 'khẳng định'),
    (10.82, 10.82, 'whip + whoosh sang B-roll xe nâng', 'chuyển cảnh'),
]

if __name__ == '__main__':
    import sys
    words = json.load(open(sys.argv[1]))
    plan = build_caption_plan(words)
    json.dump(plan, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
    n = sum(len(s['chunks']) for s in plan)
    print('chunks', n)
