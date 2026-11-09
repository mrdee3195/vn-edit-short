---
name: vn-cinematic-edit
description: Dựng video dọc 9:16 tiếng Việt theo style "người nói mở đầu + B-roll cinematic" cho mọi ngành (BĐS, F&B, spa/mỹ phẩm, giáo dục, ô tô, thời trang…) — phụ đề hiện từng từ với 3 cấp chữ (hero / nhấn / thường) do skill tự chọn theo nghĩa và hồ sơ ngành, chữ nằm sau người, thẻ UI nổi, cắt theo nhịp nhạc, chuyển cảnh nhòe, SFX riser/whoosh. Dùng khi người dùng đưa footage thô và muốn edit reel/TikTok "theo style đó".
---

# Edit video cinematic tiếng Việt (đa ngành)

Style được rút ra từ một video tham chiếu: 61 giây, 1080x1920, 30fps, nhạc khoảng 94 BPM.
Skill có 2 lớp:
- **Phần lõi** (file này): áp dụng cho mọi ngành.
- **Hồ sơ ngành** (`industries/<ngành>.md`): từ vựng, ưu tiên từ khóa, ví dụ mẫu và cấu trúc video riêng cho từng ngành.

## 0. Bắt đầu
1. Hỏi người dùng làm ngành gì.
   - Nếu đã có file `industries/<ngành>.md` thì đọc file đó.
   - Nếu chưa có, tạo hồ sơ mới theo `industries/_template.md`: xin 3–5 kịch bản/caption mẫu của người dùng, tự soạn hồ sơ nháp, rồi **đưa người dùng duyệt** trước khi dùng.
2. Xin tư liệu:
   - A-roll: người nói trước camera.
   - B-roll: cảnh minh họa theo shot list trong hồ sơ ngành.
   - Nhạc có bản quyền, kèm bộ SFX.
   - Thông tin cho thẻ UI: tên, số điện thoại, chỉ số bằng chứng xã hội.

## 1. Quy trình
1. **Chuẩn hóa clip**: dùng `ffmpeg` đưa mọi clip về 1080x1920, 30fps.
2. **Phiên âm**: dùng `faster-whisper` với model `large-v3`, `language="vi"`, bật `word_timestamps=True`.
   - Whisper trả về **từng âm tiết** chứ không phải từng từ.
   - Nếu chất lượng kém, thử PhoWhisper (model tiếng Việt của VinAI).
   - Luôn cho người dùng duyệt lại bản phiên âm, vì tên riêng và thuật ngữ ngành hay sai.
3. **Ghép âm tiết thành từ**, theo đúng thứ tự:
   1. Ghép các cụm có trong `thuật_ngữ` của hồ sơ ngành trước (VD "nhà phố", "sổ hồng", "sân vườn").
      Lý do: đã thử `underthesea`, nó tách sai các cụm này thành "nhà / phố", "sổ / hồng", "sân / vườn".
   2. Phần còn lại cho `underthesea.word_tokenize` và `pos_tag` xử lý.
   3. Mỗi từ ghép nhận timestamp bắt đầu của âm tiết đầu và kết thúc của âm tiết cuối.
4. **Chuẩn hóa số để hiển thị**:
   - "ba trăm chín mươi chín triệu" → "399 triệu"; "một trăm mét vuông" → "100 m²"; "năm mươi phần trăm" → "50%".
   - Chỉ đổi chữ hiển thị, giữ nguyên timestamp.
5. **Chia cấp từ khóa** theo mục 3, lưu vào `caption_plan.json`, rồi đưa người dùng duyệt.
6. **Tìm nhịp nhạc** bằng `librosa.beat.beat_track`, lưu vào `beats.json`, đánh dấu điểm drop.
7. **Tách người khỏi nền** ở A-roll bằng `rembg` hoặc `mediapipe` để có matte (lớp mask) cho chữ nằm sau người.
8. **Dựng timeline** theo cấu trúc trong hồ sơ ngành (nếu hồ sơ không có thì dùng mục 2).
9. **Render bằng Remotion** (mục 5), mix âm thanh theo mục 7.
10. **Kiểm tra** theo checklist ở mục 9.
11. **Xuất file**: H.264 khoảng 15 Mbps, AAC 320k, -14 LUFS.

## 2. Cấu trúc mặc định (khi hồ sơ ngành không ghi)
| % thời lượng | Đoạn | Quy tắc |
|---|---|---|
| 0–13% | Mở đầu (A-roll) | Người nói đi về phía camera. Thẻ bằng chứng xã hội hiện lên ở khoảng giây 2. Khoảng 2 giây đầu không có nhạc. |
| khoảng 13% | Drop | Riser trước điểm cắt, cắt đúng phách drop sang B-roll mạnh nhất. |
| 13–33% | Giới thiệu | Xen kẽ A-roll và B-roll, giọng nói chạy liên tục qua điểm cắt (J-cut/L-cut). |
| 33–92% | Phần trình diễn không lời | Chỉ có nhạc. Cắt theo phách: cảnh rộng 4 phách, cảnh chi tiết 2 phách, có 1 chuỗi cắt dồn dập mỗi phách một cảnh. |
| 92–96% | CTA | J-cut, kèm thẻ liên hệ. Nhạc giảm bass và nhỏ lại. |
| 96–100% | Kết | Cảnh rộng hoặc toàn cảnh, nhạc nhỏ dần trong 2.5 giây. |

## 2B. HOOK 0–10 giây: hiệu ứng hình ảnh (phần quan trọng nhất)
Đoạn hook được dựng **riêng và kỹ nhất**. Mục tiêu: người xem không có khoảnh khắc nào "đứng yên" để lướt đi.

### 2B.1 Thư viện hiệu ứng
Mọi hiệu ứng zoom đều **neo vào khuôn mặt**: dùng `mediapipe face detection` lấy tâm mặt mỗi khung hình, làm mượt chuyển động, giữ mắt ở 1/3 trên của khung.

| Mã | Hiệu ứng | Thông số | Khi nào dùng |
|---|---|---|---|
| **Z1** | **Punch-in** (zoom cắt cứng) | Nhảy tức thì từ 1.0 lên 1.2–1.25, đúng khung hình bắt đầu từ hero | Chữ hero, câu khẳng định mạnh |
| **Z2** | **Punch-out** | Nhảy tức thì từ 1.2 về 1.0 | Câu tiếp theo ngay sau Z1, để nhịp zoom vào/ra xen kẽ |
| **Z3** | **Push-in chậm** | Từ 1.0 lên 1.12 trong suốt câu, ease-in-out | Câu hỏi đặt cho người xem, câu tạo tò mò ("Bạn có biết…?") |
| **Z4** | **Zoom-out reveal** | Bắt đầu sát mặt ở 1.4, lùi về 1.0 trong 12–18 khung hình, ease-out | Lộ ra bối cảnh phía sau (kho, xưởng, công trình) |
| **Z5** | **Snap zoom** (zoom giật có nhòe) | Từ 1.0 lên 1.35 trong 4–6 khung hình, kèm nhòe zoom hướng tâm 20px, cộng whoosh | Con số gây sốc, giá, lời hứa lớn |
| **S1** | **Rung khi va chạm** | 6–8 khung hình, biên độ 12px giảm dần, xoay ±0.5° | Đi cùng sub-boom hoặc tiếng va chạm thật; dùng cùng Z1/Z5 |
| **F1** | **Lóe sáng** | 1–2 khung hình trắng, độ mờ 70% | Ngay điểm cắt, hoặc khi thẻ UI bật lên |
| **I1** | **Chèn nhanh B-roll vấn đề** | Chèn 6–12 khung hình B-roll vào giữa câu rồi quay lại A-roll | Khi người nói nêu vấn đề (tủ phồng, nứt…). Có hình minh họa lập tức |
| **P1** | **Đạo cụ trên tay** (thẻ nổi) | Thẻ ảnh hoặc video nhỏ bật lên trên lòng bàn tay, bám theo tay | Bằng chứng xã hội, sản phẩm, so sánh trước/sau |
| **SP** | **Spotlight** | Dùng matte người: làm tối nền 40% và nhòe nhẹ nền 6px, người giữ nguyên độ sáng, trong 1 câu | Câu quan trọng nhất của hook, câu chốt vấn đề |
| **FR** | **Đóng băng + cắt người** | Dừng hình 8–12 khung, người được cắt ra có viền trắng 6px, nền giảm màu, chữ hero đặt sau người | Pattern interrupt (phá nhịp) ở giây 3–6. **Tối đa 1 lần mỗi video** |
| **G1** | **Glitch / tách RGB** | 2–3 khung hình, lệch kênh màu 8px | Chỉ dùng cho câu nêu "vấn đề", và chỉ khi giọng thương hiệu trẻ trung. Mặc định tắt |
| **SR** | **Speed ramp khi đi** | Người đang bước đi: tăng 150% rồi về 100% ngay trước khi nói | Bắt đầu clip, để khung hình đầu tiên đã có chuyển động |

### 2B.2 Cách chọn hiệu ứng theo kịch bản
1. Lấy các câu thuộc 0–10s từ `caption_plan.json`. Gán **loại nhịp** cho từng câu:
   - `mở` (câu đầu tiên)
   - `vấn đề`
   - `câu hỏi`
   - `hero/khẳng định`
   - `con số`
   - `bằng chứng`
   - `lộ bối cảnh`
   - `chuyển sang drop`
2. Mỗi loại nhịp có hiệu ứng tương ứng:

| Loại nhịp | Hiệu ứng chính | Hiệu ứng phụ (tùy chọn) |
|---|---|---|
| `mở` | SR, hoặc Z4 nếu mở bằng lộ bối cảnh | Chữ hiện ngay từ khung hình 0 |
| `vấn đề` | I1 | G1 hoặc S1 nhẹ |
| `câu hỏi` | Z3 | SP |
| `hero/khẳng định` | Z1 (lần sau thì Z2, xen kẽ) | S1 + F1 nếu trùng tiếng boom |
| `con số` | Z5 | S1, cộng chữ hero đập vào màn hình (phóng 1.5→1 trong 4 khung hình) |
| `bằng chứng` | P1 | F1 |
| `lộ bối cảnh` | Z4 | – |
| `chuyển sang drop` | Push-in nhanh dần 1.0→1.15 cùng riser | Whip ra B-roll |

3. **Luật mật độ**:
   - **0–3s**: ít nhất **3 sự kiện hình ảnh**. Sự kiện là: một lần cắt, một zoom, một thẻ UI bật lên, hoặc một chữ hero xuất hiện.
   - **3–10s**: cứ 1–1.5 giây có 1 sự kiện.
   - **Sau 10s**: giảm xuống khoảng 1 sự kiện mỗi 3–4 giây (để không mệt mắt).
4. **Luật chống lạm dụng**:
   - Mỗi thời điểm tối đa 2 hiệu ứng (VD Z1 + S1).
   - Không dùng cùng một hiệu ứng zoom 2 lần liên tiếp.
   - FR tối đa 1 lần. G1 tối đa 2 lần.
   - Hiệu ứng phải rơi **đúng khung hình đầu của âm tiết** hero hoặc con số, hoặc đúng phách nhạc.
5. **Giới hạn zoom theo độ phân giải nguồn** (zoom quá mức sẽ làm hình bị mờ):
   - Nguồn 1080p: zoom tối đa 1.3.
   - Nguồn 4K: zoom tối đa 2.0.
   - Nguồn quay ngang rồi cắt dọc: tính mức zoom từ phần đã cắt.
6. **Khi chồng hiệu ứng zoom với chữ nằm sau người**: matte người phải được zoom theo cùng một phép biến đổi với video. Cách làm: zoom cả khối `nền + hero + người` cùng lúc. Phụ đề cấp 2 và 3 **không** zoom theo.
7. **Đầu ra**: lưu `hook_plan.json` gồm các mục `{t_start, t_end, effect, params, lý_do}`. Trình bày cho người dùng dưới dạng bảng **trước khi render**.

### 2B.3 Kiểm tra riêng cho hook
- [ ] Khung hình 0 đã có chuyển động và có chữ.
- [ ] Chữ hero đầu tiên xuất hiện trước giây thứ 2.
- [ ] 0–3s có ít nhất 3 sự kiện hình ảnh.
- [ ] Không zoom quá giới hạn độ phân giải; mặt người không bị cắt mất phần trán hay cằm khi zoom.
- [ ] Xem lại hook ở tốc độ 1x **có tắt tiếng**. Phần hình vẫn phải đủ hấp dẫn khi không có âm thanh, vì nhiều người lướt TikTok tắt tiếng.

## 3. Chia 3 cấp từ khóa (không phụ thuộc ngành)

### 3.1 Nguyên tắc chung
Không chia cấp theo từ loại hay theo độ nhấn giọng. Chia cấp theo **vai trò thông tin của từ trong câu quảng cáo**:

| Cấp | Vai trò | Câu hỏi để kiểm tra | Hiển thị |
|---|---|---|---|
| **1 – HERO** | **Lời hứa / kết quả / cảm xúc** chính của câu | "Nếu người xem chỉ thấy MỘT chữ trong câu này, chữ nào khiến họ dừng lướt?" | Rất to, rất đậm, ngang đầu, nằm sau người, giữ suốt cả câu |
| **2 – NHẤN** | **Bằng chứng / chi tiết cụ thể** đỡ cho lời hứa: đặc điểm, con số, đối tượng, tính từ đánh giá | "Đọc riêng các chữ cấp 2 có tóm được sản phẩm/dịch vụ không?" | Vừa, đậm |
| **3 – THƯỜNG** | Ngữ pháp, ngữ cảnh, chuyển ý, tên riêng phụ, CTA, từ lặp lại | – | Nhỏ, mảnh |

Video tham chiếu (tiếng Anh) xác nhận nguyên tắc này: người edit chọn từ nhấn theo nghĩa. Tôi đã đo âm lượng từng từ: "remodeled" và "all-brick" còn nhỏ hơn các từ xung quanh.
Với tiếng Việt, độ nhấn giọng càng không đáng tin vì cao độ đã bị dùng cho thanh điệu. Âm lượng và độ dài âm tiết chỉ dùng để phân xử khi hai từ ngang nhau.

### 3.2 Nhóm từ HERO (mọi ngành)
Mỗi hồ sơ ngành sắp xếp lại thứ tự ưu tiên của các nhóm này:
- **A. Cảm xúc / mong muốn**: mê, ghiền, thích, sướng, đã, mơ ước, xịn…
- **B. Kết quả / thay đổi**: sáng da, giảm 5 kg, tự tin, cải tạo, lột xác, tiết kiệm…
- **C. Điểm khác biệt duy nhất**: điểm khiến sản phẩm khác đối thủ (VD "nguyên căn gạch", "handmade", "không đường").
- **D. Con số gây sốc**: chỉ khi con số **chính là** điểm hút người xem (giá rẻ bất ngờ, kết quả lớn, thời gian ngắn).

### 3.3 Quy tắc riêng cho tiếng Việt
1. **Đơn vị là TỪ chứ không phải âm tiết.** Không bao giờ tách "sổ / hồng", "cải / tạo". Hero tối đa khoảng 2–3 âm tiết và khoảng 10 ký tự mỗi dòng.
2. **Danh sách khóa cứng ở cấp 3**:
   - Đại từ: tôi, mình, bạn, anh, chị, em, mọi người, các bạn.
   - Giới từ, liên từ: của, và, với, cho, thì, mà, là, nếu, nhưng, để.
   - Lượng từ, chỉ định: các, những, một, cái, con, căn (khi là loại từ), này, đó, kia.
   - Phó từ thời gian: đã, đang, sẽ, vừa, mới.
   - Từ bị động: được, bị.
   - **Tiểu từ cuối câu**: nha, nhé, ạ, á, luôn, đó, nè, ha, hen.
   - Theo nhãn POS của `underthesea`: P, E, C, T, L, Nc, và R (trừ từ mức độ).
3. **Từ mức độ đi kèm hero** (siêu, cực, cực kỳ, rất, quá trời, xịn xò, đỉnh) thành **kicker**: chữ nhỏ đặt sát hero. Không tự nó làm hero.
4. **Kicker và sub xác định theo vị trí, không theo từ loại.** Tiếng Việt đặt bổ ngữ **sau** danh từ, ngược với tiếng Anh. Từ đứng ngay trước hero thành kicker (đặt phía trên bên trái), từ đứng ngay sau thành sub (đặt phía dưới).
   - VD: "phòng khách **rộng** thênh thang" → "phòng khách" là kicker, "rộng" là hero, "thênh thang" là sub.
5. **Từ tiếng Anh chen vào câu** (view, full nội thất, sale, deal, combo, skincare) được xử lý như từ nội dung bình thường. Viết đúng cách người dùng hay viết.
6. **Từ lặp lại**: lần đầu xét bình thường, các lần sau hạ một cấp.
7. **Tên riêng** (địa danh, thương hiệu, tên người) để cấp 3. Trừ khi hồ sơ ngành ghi tên đó chính là điểm bán (VD "Phú Quốc", "Vinhomes").
8. **Câu CTA** để cấp 3, vì thẻ liên hệ đã làm nhiệm vụ nhấn mạnh.
9. **Tuân thủ quảng cáo (Luật Quảng cáo VN)**:
   - Các từ tuyệt đối như "nhất", "số 1", "duy nhất", "100%", "khỏi hẳn", "cam kết" **không được tự động làm hero**.
   - Gắn cờ ⚠ trong bảng duyệt để người dùng xác nhận có căn cứ.
   - Ngành y tế, thẩm mỹ, thực phẩm chức năng cần chặt hơn (xem hồ sơ ngành).

### 3.4 Thuật toán
**Bước A – Code gắn nhãn sơ bộ** (`scripts/tag_words.py`): mỗi từ nhận các trường sau:
- `pos`: nhãn từ loại của underthesea.
- `stop`: có trong danh sách khóa cấp 3 hay không.
- `lex`: nhãn theo từ điển của hồ sơ ngành (`emotion` / `result` / `feature` / `differentiator` / `number` / `place` / `brand` / `intensifier`).
- `nth`: lần xuất hiện thứ mấy của từ.
- `shot`: cảnh là A-roll hay B-roll.
- `db_rel`: âm lượng so với trung bình câu (chỉ để tham khảo).
- `compliance`: có phải từ tuyệt đối hay không.

**Bước B – Claude chấm theo nghĩa.** Mỗi lần xử lý một câu, prompt gồm: rubric ở mục 3.1–3.3, thứ tự ưu tiên của ngành, **5–10 câu mẫu đã gán cấp lấy từ hồ sơ ngành**, và câu cần chấm kèm nhãn sơ bộ.
Kết quả trả về:
`[{"w","start","end","tier":1|2|3,"role":"hero|kicker|sub|caption","why":"..."}]`
Trường `why` là một lý do ngắn, hiển thị khi người dùng duyệt.

**Bước C – Code kiểm tra lại** (bắt buộc):
- Cấp 1 + cấp 2 không quá 30% tổng số từ (hồ sơ ngành được chỉnh tỉ lệ này).
- Mỗi câu tối đa 1 hero. Hai hero cách nhau ít nhất 2 giây. Cả video có 3–5 hero, dồn vào đoạn mở đầu.
- Hero chỉ đặt trên cảnh A-roll có matte người. Ở B-roll thì hero hạ xuống cấp 2.
- Không có từ nào trong danh sách khóa cứng ở cấp 1 hoặc 2.
- Hero không chứa từ tuyệt đối khi chưa được người dùng xác nhận.
- Nếu vi phạm: hạ cấp từ yếu nhất, rồi chạy kiểm tra lại.

**Bước D – Người dùng duyệt.**
- In bảng gồm: câu, chữ cấp 1, chữ cấp 2, cờ ⚠.
- Người dùng sửa xong thì lưu các sửa đổi vào mục `ví_dụ` của hồ sơ ngành, để lần sau skill chọn đúng hơn.

### 3.5 Ví dụ đa ngành
Ký hiệu: **[1]** là hero, *[2]* là nhấn, chữ thường là cấp 3, (k) là kicker.

| Ngành | Câu |
|---|---|
| BĐS | Căn *nhà phố 3 tầng* này vừa được (k) **[1] cải tạo** toàn bộ, *sân vườn* *100 m²*, đã có *sổ hồng* luôn nha |
| F&B | Nếu bạn (k) **[1] mê** *phô mai* thì món *pizza* này sẽ làm bạn *ghiền* luôn |
| Spa | Chỉ sau *3 buổi*, làn da *sạm nám* của chị đã (k) **[1] sáng** *rõ rệt* |
| Giáo dục | Khóa học giúp bạn *giao tiếp* *tiếng Anh* **[1] tự tin** chỉ trong *90 ngày* |
| Ô tô | Mẫu *SUV 7 chỗ* này chỉ (k) **[1] 6 lít** / *100 km* (con số chính là điểm hút → hero nhóm D) |
| Thời trang | Chiếc *váy linen* này mặc lên **[1] tôn dáng** cực, đi *biển* hay đi *làm* đều hợp |

## 4. Thông số chữ (đã chỉnh cho tiếng Việt)
- **Font bắt buộc hỗ trợ đầy đủ dấu tiếng Việt**: Be Vietnam Pro, Inter, Montserrat, SVN-Gilroy, SF Pro.
  - Cấp 3: độ đậm 300. Cấp 2: 700. Cấp 1: 800–900.
  - Phải test cả các chữ "Ữ Ặ Ỷ" trước khi render.
- **Khoảng cách dòng**: tối thiểu 1.15. Không dùng 0.9 như bản tiếng Anh, vì dấu trên và dấu dưới sẽ bị cắt hoặc chồng lên dòng khác.
- **Khoảng cách chữ**: hero -1%. Không ép chặt hơn vì dấu sẽ dính nhau.
- **Kích thước** (khung rộng 1080px): hero 110–150px, cấp 2 56–64px, cấp 3 44–50px.
- **Màu**: trắng #FFFFFF, đổ bóng nhẹ `0 2px 12px rgba(0,0,0,.35)`.
  - Tùy chọn cho hero là tên vật liệu hoặc màu: dùng chế độ hòa trộn overlay để chữ ăn màu phía sau.
- **Dải dọc trong khung**:
  - Hero ở 18–30% chiều cao khung.
  - Cấp 2 và 3 ở 40–48%.
  - Không đặt chữ ở 20% dưới cùng và 12% bên phải.
- **Chia cụm**:
  - Mỗi lần hiện 1 từ (có thể nhiều âm tiết) hoặc 2 từ ngắn.
  - Chữ xuất hiện tại `start - 2 khung hình`.
  - Hero giữ nguyên suốt cả câu.
- **Hiệu ứng**:
  - Xuất hiện trong 5–6 khung hình: độ mờ 0→1, nhòe 12→0px, trượt lên 20→0px, thu nhỏ 1.06→1.
  - Khi đang hiện: trôi lên khoảng 8px/giây.
  - Biến mất: mờ dần trong 3 khung hình.
- **Thứ tự lớp**: `nền → hero → người (đã mask) → kicker/sub/phụ đề`.
- **Phần trình diễn không lời**: không có chữ.

## 5. Component Remotion
- **`<KineticWord>`**, **`<CaptionTrack plan>`**, **`<SubjectMatte>`**: như mô tả ở mục 4.
- **`<ProofCard>`**: thẻ bằng chứng xã hội.
  - Hình dạng: thumbnail 9:16, bo góc 18px, nhãn ở đáy (VD "600K 👀", "5.000+ khách", "4.9★").
  - Xuất hiện: 1 khung hình trắng lóe lên, rồi phóng 0.6→1.0 trong 4 khung hình (spring).
  - Bám theo bàn tay người nói.
- **`<ContactPill>`**: thẻ liên hệ kiểu "Dynamic Island".
  - Hình dạng: viên thuốc 520x84px, nền kính mờ.
  - Xuất hiện: bắt đầu là chấm nhỏ, kéo dài thành thẻ trong 8 khung hình, rồi chữ hiện ra.
  - Nội dung mặc định: tên + số điện thoại/Zalo.
- **`<WhipTransition>`**: chuyển cảnh nhòe, thông số ở mục 6.

## 6. Chuyển động và chuyển cảnh
- **Chuyển cảnh nhòe (whip)**:
  - 6–10 khung hình cuối cảnh trước: nhòe tăng 0→60px, phóng to 1.0→1.08.
  - Đúng khung hình cắt: 1 khung nhòe xoáy tròn.
  - 5–6 khung hình đầu cảnh sau: nhòe giảm 60→0.
- **Đổi tốc độ**: B-roll quay 60fps; chạy 100%, chậm xuống 30% ở chi tiết đẹp, tăng lên trên 200% khi vào cú quay nhanh.
- **Đẩy máy**: cảnh ít chuyển động thì zoom số 2–4%.
- **Chuỗi cắt dồn dập**: 6–8 cảnh chi tiết, mỗi cảnh 1 phách.
- **J-cut**: giọng và chữ của đoạn sau bắt đầu sớm hơn hình 6–12 khung hình.
- **Màu**: mặc định ấm và rực (saturation +15–20, tăng contrast theo đường cong chữ S). Hồ sơ ngành có thể đổi (VD spa dùng tông trong và sáng, ô tô dùng tông lạnh và tương phản cao).

## 7. Âm thanh và SFX
- **Giọng nói**:
  - Cắt tần số dưới 80Hz, nén 3:1, khử tiếng xì, khoảng -16 LUFS.
  - Nhạc tự giảm 10 dB khi có giọng nói (attack 50ms, release 300ms).
- **Bản đồ nhạc theo thời gian**:
  1. Khoảng 2 giây đầu không nhạc.
  2. Sub-boom cùng lúc thẻ bằng chứng hiện lên.
  3. Riser + whoosh dài 0.6–0.8s, đỉnh rơi đúng điểm cắt vào drop.
  4. Riser thứ hai ở lần chuyển phần lớn tiếp theo.
  5. Đến CTA: lọc thấp khoảng 800Hz, giảm 8 dB.
  6. Nhỏ dần 2.5 giây ở cảnh kết.
- **SFX nhỏ**:
  - Pop khi thẻ bằng chứng hiện; swipe khi thẻ liên hệ hiện.
  - Whoosh rất nhỏ (-18 dB) ở chuyển cảnh nhòe.
  - SFX nằm chìm dưới nhạc.
- **Chuẩn đầu ra**: -14 LUFS, đỉnh không quá -1 dBTP.

## 8. Shot list
Lấy từ hồ sơ ngành. Nguyên tắc chung:
- Người nói vừa đi vừa nói về phía camera, có động tác ngửa lòng bàn tay ở câu đầu.
- B-roll quay góc siêu rộng, góc thấp, 60fps, có vật ở tiền cảnh, luôn di chuyển.
- Có 1 cảnh toàn cảnh để kết.

## 9. Checklist kiểm tra
- [ ] Bản phiên âm đã được người dùng duyệt (tên riêng, thuật ngữ, số).
- [ ] Không có từ nào bị tách giữa chừng; số đã đổi sang dạng chữ số.
- [ ] `caption_plan.json` qua hết các kiểm tra ở mục 3.4 Bước C và đã được duyệt; không còn cờ ⚠ chưa xử lý.
- [ ] Dấu tiếng Việt không bị cắt ở mọi cỡ chữ.
- [ ] Mọi từ hiện lên lệch với giọng nói không quá 3 khung hình.
- [ ] Không có chữ trong vùng giao diện TikTok.
- [ ] Mọi điểm cắt ở phần trình diễn lệch phách không quá 1 khung hình; đỉnh riser rơi đúng điểm cắt.
- [ ] Viền mask quanh tóc không nhấp nháy.
- [ ] Số điện thoại hiện đủ để đọc ít nhất 1.5 giây.

## 10. Bài học từ lần test 1 (Thuận Phát – MDF, 11/2026)
Bộ render đã chạy thật bằng **Python + OpenCV + PIL** (`scripts/render.py`, `scripts/audio.py`). Đây là cách mặc định, chạy được trên máy không có GPU: video 43 giây 4K render mất khoảng 2 phút. Remotion chỉ là phương án thay thế. Riêng `scripts/plan.py` là file kế hoạch của từng video, mỗi video mới phải viết lại.

1. **Tên thương hiệu:** Whisper nghe sai ("Thượng Pháp" → "Thuận Phát"). Đọc tem, logo có trong khung hình, hoặc hỏi người dùng.
2. **Cắt sau khi kiểm tra chữ cuối:** trước khi chốt điểm cắt audio, kiểm tra `end` của từ cuối (lần test có "formaldehyde" bị cắt cụt). `plan.py` phải assert không có từ nào vượt ra ngoài đoạn.
3. **Màn hình đen:** chạy `blackdetect` để tìm và bỏ các khung hình đen có sẵn trong footage.
4. **Zoom theo kích thước mặt, không theo số cố định:**
   - Zoom nền = 0.16 / chiều_cao_mặt (để mặt chiếm khoảng 16% chiều cao khung).
   - Punch-in = zoom nền × 1.25; snap = zoom nền × 1.3.
   - Đặt mặt ở khoảng 42% chiều cao khung. Phụ đề đặt ở khoảng 60% (trên ngực).
5. **Lớp chữ không zoom theo video.** Chỉ video và matte người zoom cùng một phép biến đổi; chữ đứng yên trên màn hình.
6. **Quy tắc che chữ hero:** kiểm tra **từng khung hình** xem người che bao nhiêu % chữ. Nếu quá 12% thì đẩy chữ lên (trượt tối đa 30px/khung, chỉ đi lên, không đi xuống). Chữ kicker đi theo hero.
7. **Câu so sánh "A hay B":** cho phép 2 hero xếp chồng (A / "hay là" / B). Đây là ngoại lệ của luật "1 hero mỗi câu".
8. **Chữ cấp 3 dùng Bold 48 thay vì Light** khi nền phía sau rối (kệ hàng, kho). Đổ bóng đậm hơn (alpha 170).
9. **Đoạn drop:** làm tối khung hình 28% để chữ slam trắng nổi lên. Nhãn trên màn split-screen dùng thẻ nền đỏ thương hiệu.
10. **Giọng nói từ 2 nguồn thu khác nhau** (A-roll và VO ở B-roll) có âm lượng chênh nhau. Dùng `dynaudnorm` trước, rồi `loudnorm`.
11. **Nhạc tổng hợp** trong `audio.py` chỉ là nhạc tạm. Bản đăng thật phải thay bằng nhạc có bản quyền, bám đúng mốc thời gian.

## 11. Cập nhật sau góp ý v1 (người dùng duyệt)
**Chữ:**
- Chữ cấp 2 (nhấn) **IN HOA toàn bộ**, ExtraBold 64.
- Chữ cấp 3 (thường) dùng **Light 52** (mảnh), đổ bóng đậm (alpha 190, blur 5) để vẫn đọc được trên nền rối. ExtraLight quá mờ trên nền kho/kệ, không dùng.
- Hero giữ chữ thường, Black 150–160.

**Hai chế độ phụ đề, dùng xen kẽ:**
| Chế độ | Cách hiện | Dùng khi |
|---|---|---|
| **Từng cụm** | 1–2 từ mỗi lần, thay nhau | Hook 0–10s, câu có hero, câu CTA |
| **Cả câu** (`PHRASE`) | Chữ thường và chữ IN HOA trên **cùng một khung**, từng từ hiện dần theo lời, giữ đến hết khối | Câu giải thích có ≥ 2 từ cấp 2, ngoài hook, không chứa hero. Khoảng 30–50% số câu |

Quy tắc khối "cả câu":
- Tối đa 2 dòng, rộng 900px, cách dòng 82px, căn trên cùng một baseline.
- **Các từ IN HOA liền nhau không được tách dòng** (VD "KIỂM SOÁT CHẤT LƯỢNG").
- Ưu tiên ngắt khối sau dấu phẩy.
- Từ hiện vào trong 6 khung hình (mờ → rõ, trượt lên 14px), không trôi lên khi đang hiện.

**SFX mượt (bỏ tiếng "bụp bụp"):**
- **Không có kick dưới lời nói.** Kick và clap chỉ dùng trong bar drop. Phần còn lại là pad, bass ngân dài và shaker nhẹ.
- Không dùng boom có tiếng click. Thay bằng `impact_soft` (attack 18ms, sub 38Hz, đuôi 1.8s), và chỉ dùng ở drop, VS, hero cuối và end card.
- Trước chữ hero dùng **swell** (tiếng hút vào kiểu reverse-reverb), kết thúc đúng tại chữ, thay cho boom.
- Pop thay bằng **tick** kiểu kính (1760Hz, rất nhỏ).
- Toàn bộ SFX và nhạc đi qua reverb chung (IR 1.6s, wet 25–28%).
- Ducking nhả chậm hơn (450ms) và được làm mượt.
- Kiểm tra: không được có bước nhảy âm lượng > 9 dB trong 10ms tại các mốc SFX.

**Thẻ liên hệ** hạ xuống y = 1330 để không đè phụ đề (y = 1160).

## 12. Thư viện SFX (`scripts/sfx_lib.py`)
Thư viện có 2 nguồn. **File thật được ưu tiên hơn tiếng tổng hợp nếu trùng tên.**

1. **SYNTH**: 12 tiếng tổng hợp bằng code, luôn có sẵn, không bản quyền.
   - impact_soft · sub_drop · swell · riser · whoosh · swipe · tick · card_flip · ding · wood_knock · shutter · glitch
   - Nghe thử: `out/sfx_library_demo.mp4`
2. **FILES**: thư mục `sfx/` + file `sfx/catalog.json`.
   - Mỗi tiếng ghi: `name, file, hit, gain_db, use, license`.
   - `hit` = số giây tính từ đầu file đến khoảnh khắc phải rơi đúng khung hình (đỉnh whoosh, lúc ván chạm sàn…).
   - Dùng `place_hit()` để đặt **điểm hit** (không phải đầu file) đúng vào điểm cắt hoặc chữ.

Thứ tự ưu tiên chất lượng:
1. Âm thanh thật của sản phẩm do khách tự thu (gõ ván, cưa, xe nâng, nước).
2. Bộ SFX có bản quyền thương mại (Artlist, Epidemic, Envato…).
3. Tiếng tổng hợp.

Mọi file đều phải ghi rõ `license`. Không dùng SFX không rõ nguồn.

**Bảng ghép hiệu ứng hình → SFX:**
| Hiệu ứng hình | SFX |
|---|---|
| Z1 punch-in | tick (hoặc wood_knock với ngành gỗ) |
| Z5 snap zoom | whoosh + ding |
| Z4 reveal | sub_drop |
| S1 rung | impact_soft hoặc wood_knock |
| F1 lóe sáng / FR đóng băng | shutter |
| G1 glitch | glitch |
| P1 thẻ nổi trên tay | card_flip |
| Mục danh sách | tick hoặc card_flip |
| Thẻ liên hệ | swipe |
| Trước chữ hero | swell |
| Dẫn vào drop | riser + impact_soft |

## 13. Mẫu hook có tên (từ buổi thử 5+1 kiểu, `scripts/hooklab.py`)
Người dùng (Thuận Phát) chọn: **ĐÓNG BĂNG**, **NĂNG LƯỢNG CAO**, **THĂM DÒ**, và đang thử **TAN BIẾN**.

| Mẫu | Ý chính | Hợp khi |
|---|---|---|
| ĐỐI ĐẦU | Khung hình 0 là split-screen A vs B, mặt người nói trong khung tròn rồi nở ra full | Câu hook dạng "A hay B" |
| **ĐÓNG BĂNG** ✓ | Dừng hình đúng từ khóa, cắt người có viền trắng, nền xám tối, nhãn đỏ bật lên, rồi chớp 3 cảnh B-roll trong khoảng lặng | Muốn phá nhịp lướt |
| **NĂNG LƯỢNG CAO** ✓ | Zoom giật + chữ đập vào mỗi từ khóa, rung, glitch câu hỏi, từ khóa nền đỏ, màn VS | Người xem trẻ, nhịp nhanh |
| ĐIỆN ẢNH | Mở bằng cận cảnh sản phẩm có viền đen kiểu phim, zoom-out từ sản phẩm trên tay ra mặt người nói | Thương hiệu cao cấp, khách B2B |
| **THĂM DÒ** ✓ | Bảng bình chọn kiểu TikTok, các phương án sáng lên theo lời nói, kết bằng gợi ý bình luận. Không hiện % giả | Muốn tăng bình luận |
| **TAN BIẾN** (thử) | Nói xong hook thì zoom-out, người tan thành bụi, còn lại phòng trống, câu sau là VO trên B-roll | Chuyển từ "người" sang "sản phẩm" |

### Kỹ thuật TAN BIẾN
1. **Nền sạch** (`plate.py`):
   - Lấy mẫu khung hình từ mọi take A-roll, mỗi 0.4 giây.
   - Tách người (rembg), nới mask thêm phần tóc và bóng.
   - Căn mỗi khung về khung tham chiếu bằng ORB + homography, chỉ dùng điểm đặc trưng trên nền.
   - Lấy trung vị theo từng điểm ảnh, bỏ qua vùng có người.
2. **Vá phần chưa bao giờ lộ** (`lama_fill.py`):
   - Model LaMa ONNX (Carve/LaMa-ONNX trên HuggingFace, 512x512), cắt vùng vuông chồng lên nhau quanh lỗ.
   - Trong lần test, vùng sau lưng người nói chưa bao giờ lộ ra (khoảng 11% khung). Thuật toán Telea để lại vết nhòe; LaMa vá kệ và tủ liền mạch.
3. **Hiệu ứng tan:**
   - Đóng băng đúng khung tham chiếu, rồi snap zoom-out ra khung rộng (zoom ≈ 1.55).
   - Mỗi điểm ảnh có thời điểm tan τ theo hướng quét (dưới-trái → trên-phải) cộng nhiễu.
   - Viền ấm mảnh (0.03 giây) ở mép đang tan.
   - Hạt bụi lấy từ điểm ảnh của người (lưới 3px), màu sáng lên, bay lên-phải có dao động, sống 0.55–1.1 giây, mỗi hạt vẽ 2x2 px.
   - SFX: riser → whoosh (zoom-out) → `dust` + `sub_drop`.
4. **Điều kiện:**
   - Máy quay gần như cố định, và có đủ khung hình người đứng lệch chỗ.
   - **Tốt nhất: thợ quay ghi thêm 3 giây phòng trống** (nền sạch thật). Khi đó không cần ghép và vá.
