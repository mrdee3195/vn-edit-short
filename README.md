# vn-cinematic-edit

Skill cho Claude Code: dựng video dọc 9:16 tiếng Việt theo style "người nói mở đầu + B-roll cinematic", dùng được cho nhiều ngành.

- `SKILL.md`: hướng dẫn chính (quy trình, 3 cấp từ khóa, hook, chữ, chuyển động, âm thanh, checklist, bài học từ các lần test).
- `industries/`: hồ sơ ngành (mẫu, bất động sản, vật liệu xây dựng – ván gỗ).
- `scripts/`: các công cụ đã chạy thật.
  - Phiên âm: `tw_vi.py`
  - Dò mặt: `face_track.py`
  - Tách người khỏi nền: `matte.py`
  - Render: `render.py`, `plan.py`
  - Âm thanh: `audio.py`, `sfx_lib.py`
  - Thử hook: `hooklab.py`
  - Nền sạch: `plate.py`, `lama_fill.py`
- `sfx/catalog.json`: mẫu khai báo file SFX thật.

## Cài đặt
Chép thư mục này vào `.claude/skills/vn-cinematic-edit/` (trong project) hoặc `~/.claude/skills/vn-cinematic-edit/`.

Phụ thuộc Python:
```
pip install faster-whisper underthesea opencv-python-headless pillow rembg onnxruntime scipy soundfile numpy
```

Cần tải riêng:
- Font Be Vietnam Pro (Google Fonts).
- Model YuNet: `opencv/face_detection_yunet` trên HuggingFace.
- Model LaMa ONNX: `Carve/LaMa-ONNX` trên HuggingFace.

Đường dẫn `/home/user/project/...` trong các script là của lần test đầu, cần sửa lại theo máy của bạn.
