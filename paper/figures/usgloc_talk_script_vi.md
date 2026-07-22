### 1. Trang tiêu đề *(~40 giây)*
deeply technical: when a robot tells you where it is — *should you believe it?
Câu trả lời  "minh đang ở đâu" không phải là một điểm, mà là một distribution.

### 2. minh có dám bay trên nó không? *(~2 phút)*
Nhung SOTA visual localization systems đều có chung một điểm: chúng luôn luôn, mọi lúc,
confident một trăm phần trăm. 

Cho chúng xem một góc nhìn hoàn hảo, đủ sáng, của một căn phòng chúng đã map hôm qua: chúng xuất ra một pose. 
Cho chúng xem một góc nhìn bị che khuất một nửa, của một căn phòng mà ai đó đã dọn
lại toàn bộ đồ đạc: chúng vẫn xuất ra một pose. *Cùng một kiểu câu trả lời,
cùng một mức độ tự tin.

Vậy đây là toàn bộ buổi seminar gói gọn trong một câu: làm sao để một bản
đồ 3D thừa nhận được những gì nó *không* biết — và làm sao biến sự trung
thực đó thành một con robot an toàn hơn?


### 3. Dàn ý 

---

### 4. Visual localization — nhập môn *(~2.5 phút)*
Bài toán: một query image, một prior/pre-built map, ước lượng đầy đủ camera pose
6 bậc tự do (6-DoF). Tên cổ điển của nó là bài toán "kidnapped robot":
robot tỉnh dậy ở một nơi nó đã từng đến, và phải tự tìm ra mình đang ở đâu.

Ứng dụng ở đâu? Kính AR neo lại nội dung ảo giữa các
phiên sử dụng; robot và drone phục hồi sau khi mất tracking; xe tự hành
định vị ở mức làn đường khi GPS mất tín hiệu trong hầm.

Công thức chủ đạo không thay đổi trong cả một thập kỷ: một giai đoạn *coarse* — place recognition hoặc một learned
regressor đưa minh đến gần đúng vị trí — và một giai đoạn *fine* — căn chỉnh
image với map. 


### 5. Từ point cloud đến Gaussian Splatting *(~2.5 phút)*
Ba thế hệ. Thứ nhất, sparse point cloud từ structure-from-motion — match
keypoint, PnP solver; vẫn là một baseline mạnh. 
Thứ hai, NeRF, năm 2020:
scene được biểu diễn như một neural network; ảnh chân thực nhưng render một
frame mất vài giây — rất khó chịu khi localizer của minh cần hàng trăm lần
render. 
Thứ ba, Gaussian Splatting, năm 2023: scene là vài triệu quả
ellipsoid tường minh nhỏ — vị trí, hình dạng, opacity, màu sắc — được
render bằng cách chiếu và alpha-blending, chân thực *ở real-time*.

 Đây là toàn bộ renderer: mỗi Gaussian đóng góp
màu của nó, có trọng số theo opacity của nó và theo việc có bao nhiêu ánh
sáng lọt qua từ những gì phía trước nó. 

Hai đặc tính quan trọng với chúng
tôi. Nó differentiable theo camera pose — nên minh có thể dùng gradient
descent để tìm về đúng pose. Và bất kỳ quantity nào gắn với từng splat
cũng có thể "đi nhờ" như một color channel phụ.


### 6. Uncertainty — nhập môn *(~2.5 phút)*
Hai loại "không biết", và việc minh có loại nào là rất quan trọng.
*Aleatoric*: nhiễu vốn có trong thế giới — sensor noise, motion blur, một
bức tường không có texture. minh không thể sửa nó bằng cách thu thêm dữ
liệu. 
*Epistemic*: những lỗ hổng trong kiến thức của model — một viewpoint
chưa từng được map, hoặc, một scene đã *thay
đổi* sau khi mapping. Reducible — nếu minh biết nó nằm ở đâu.
 
Và một phép thử giữ cho mọi người trung thực: calibration. Một người dự
báo thời tiết nói bảy mươi phần trăm khả năng mưa thì nên bị ướt vào đúng
bảy mươi phần trăm những ngày như vậy. Với pose: một credible region chín
mươi lăm phần trăm nên chứa true pose chín mươi lăm phần trăm số lần.
Không nhiều hơn — đó là sự thận trọng lãng phí. Không ít hơn — đó là
trường hợp nguy hiểm. *

→ *Vậy làm sao để một network nói thật rằng "tôi không biết"? Đó chính là
Evidential Deep Learning.*

### 7. Evidential Deep Learning trong ba phút *(~2.5 phút)*
Hãy nghĩ về điều mà softmax không thể làm được. Probability phải cộng lại
bằng một. Vậy nên sự vô tri hoàn toàn — "tôi chưa từng thấy thứ gì như thế
này" — và sự ambiguity thực sự — "tôi đã thấy nó cả ngàn lần, nó là một cú
tung đồng xu" — cho ra *cùng một output*. Không có cách nào để nói "tôi
không có ý tưởng gì".

Nước đi kiểu evidential: thay vì probability, network xuất ra *evidence* —
về cơ bản là các số đếm không âm. Từ evidence, minh hình thành một opinion:
belief cho mỗi class, cộng thêm một mass phụ, gọi là **vacuity** — mass
"tôi không biết" một cách tường minh. Một công thức cần nhớ: vacuity bằng
C chia S — số class chia cho tổng evidence. Không có evidence: vacuity
bằng một. Có núi evidence đồng thuận: vacuity bằng không.

**[chỉ vào tam giác]** Về mặt hình học: softmax sống trên cạnh đáy của tam
giác này — vacuity luôn đồng nhất bằng không. Evidential learning giải
phóng model để di chuyển lên trên. Thêm một dimension duy nhất, và "chưa
từng thấy cái này" so với "đã thấy nó, nhưng nó ambiguous" trở thành *hai
output khác nhau*.

### 8. Dempster–Shafer fusion trong ba phút *(~2 phút)*
Giờ thì, map được xây dựng từ rất nhiều observation — vậy nên chúng ta cần
*fuse* các opinion lại. Dempster's rule: sự đồng thuận nhân belief lên, và
vacuity của mỗi bên cho phép bên kia được "lên tiếng". **[đi qua bảng số
liệu]** Hai sensor đều nói "chair", mỗi cái có một chút vacuity — fuse
lại: belief 0.94, vacuity giảm xuống còn 0.06. Sự đồng thuận làm tăng
confidence. Hoàn toàn đúng đắn khi fuse nhiều view của một object ổn định.

**[khối "cliffhanger" — nói chậm lại]** Nhưng đây là một câu hỏi cần giữ
trong đầu: điều gì xảy ra nếu hai source *confidently disagree* với nhau?
Sensor A: chắc chắn là chair. Sensor B: chắc chắn là box. Trực giác nói
rằng opinion sau khi fuse nên hét lên "có gì đó không ổn ở đây". Dempster's
rule lại nói... một confident compromise. Đó là Zadeh's paradox, nó đã năm
mươi tuổi rồi, và nó sẽ quay trở lại ở nửa sau bài nói này như một *real
bug chúng tôi đã gặp phải*. Hãy nhớ nó — vì "confident disagreement giữa
các view" chính xác là dấu hiệu thống kê của một object đã bị move.

---

## Phần: Bài toán — 4 phút

### 9. Motivation I: tình huống cụ thể *(~2 phút)*
Bây giờ là bối cảnh cụ thể. Một drone đang relocalize trong một căn phòng
"có người ở". Map là một semantic Gaussian-splatting map, được train một
lần. Giữa lúc mapping và lúc bay: màu xanh lá là những gì không đổi, màu cam là những gì đã bị
move. Và pose estimate không được một con người tiêu thụ — nó đi vào một
Kalman filter, được fuse với IMU, hàng trăm lần mỗi phút. Câu hỏi của
filter không bao giờ chỉ là "ở đâu" — mà là "ở đâu, *và với covariance
nào*". 

những vùng đánh lừa việc localization chính là
những vùng đã thay đổi — và map đã nhìn thấy đủ để *biết* điều đó. Map
ngày nay chỉ đơn giản là không có "từ vựng" để nói ra điều đó.

### 10. Motivation II: điều gì sai khi thiếu uncertainty *(~2 phút)*
Ba kiểu failure khi thiếu uncertainty. 
Một: không có covariance nghĩa là
filter phải giả định một fixed noise model — nên nó *overconfident đúng
vào lúc scene ambiguous, occluded, hoặc đã thay đổi*, tức là đúng lúc
accuracy thực sự suy giảm. 
Hai: giải pháp hiện có cho việc scene thay đổi
là một hard mask — một LLM quyết định object nào là "reliable" — nhị phân,
giữ lại hoặc loại bỏ, và bản thân bộ chọn lọc đó cũng có những failure case
đã được ghi nhận. 
Ba: khi một căn phòng có hai góc tương tự nhau, error
trở nên *multi-modal* — "tôi ở đây hoặc ở kia" — và cả một point lẫn một
single Gaussian đều không thể diễn đạt điều đó.

**[cột bên phải]** Vậy nên có ba requirement: calibrated, graded, và
multi-modal. Mọi thứ trong USGLoc được thiết kế ngược lại từ ba yêu cầu
này.

---

## Phần: Công trình liên quan — 7 phút (kể cả slide "khoảng trống")

*(Đây là các slide dạng bảng — ĐỪNG đọc chúng. Đi qua một hàng cho mỗi
slide, để khán giả tự lướt phần còn lại, chốt lại ở khối "câu chốt".)*

### 11. Related work I: visual localization *(~1.5 phút)*
Bốn loai phương pháp, và với mỗi loai, minh muốn các minh nhìn vào cột cuối cùng
— limitation. Pose regression: nhanh, nhưng luon overconfident khi ra
ngoài phạm vi các training view. 
Scene coordinate regression — những "ông
vua" về accuracy — nhưng chú ý ACE-G: nó *có* uncertainty theo từng
coordinate, và chỉ dùng nó để lọc bỏ các correspondence xấu một cách nội
bộ. 
Các feed-forward matcher như MASt3R: cũng có confidence — cũng chỉ
được dùng nội bộ. 
Các phương pháp semantic: object và scene graph, nhưng
không có metric distribution. **[câu chốt]** Pattern chung, xuyên suốt cả
bốn họ: uncertainty là một *ingredient*, không bao giờ là *output*. Không
ai trao cho robot một calibrated covariance.

### 12. Related work II: rendering-based localization *(~2 phút)*
Các NeRF refiner: chính xác nhưng chậm. 
Particle filter — Loc-NeRF, MCLoc — cái này đau nhất: chúng duy trì
một belief thực sự về pose, hàng trăm hypothesis... rồi chỉ báo cáo mỗi
cái tốt nhất. Belief chết trước khi đến được output. 
Gaussian-splatting Wave — 2024-2026 dat sub-centimeter accurate — tất cả đều là point output, tất cả đều
giả định scene không bao giờ thay đổi. 

Và công trình gần nhất với chúng tôi, UGS-Loc tại CVPR năm nay: nó render Fisher-information uncertainty
map và dùng chúng để guide việc search — thực sự khéo léo — nhưng
uncertainty đó là post-hoc, thuần túy hình học, và internal. Output: một
pose duy nhất. Không covariance, không calibration, không semantics.
**[hàng semantic]** Và cặp phương pháp semantic — SGLoc, Chu & Kim — robust
với scene change, nhưng qua cơ chế cổng LLM nhị phân đã nói ở trên.

### 13. Related work III: uncertainty methods *(~1.5 phút)*
Và chính cộng đồng nghiên cứu về uncertainty. 
Bayesian pose regressor:
distribution thực sự, nhưng chưa calibrated, giới hạn về accuracy.

Probabilistic PnP và conformal method: chặt chẽ — nhưng cho *object*,
không phải cho cả scene. 
Wu và Campbell chỉ ra rằng pose error không theo
Gaussian và cần một mixture — trong bài toán 2D retrieval; chúng tôi nâng
chính xác insight đó lên 6-DoF. 
Evidential learning đã tiếp cận được đến
scene coordinate regression. Và cộng đồng mapping — dòng BKI, E²-BKI,
continuous semantic splatting — xây dựng những map thực sự trustworthy
nhưng *không bao giờ localize bất cứ điều gì*.


### 14. Ô trống trong design space *(~2 phút)*
Trên-trái: uncertainty-aware mapping. Dưới-phải: accurate
point-estimate localization. Wu & Campbell: có distribution, nhưng chỉ
2D. UGS-Loc: ở phía localization, có uncertainty bên trong, nhưng output
là một point. Ô trên-phải — calibrated SE(3) localization dựa trên một
splatting map — là, theo một cuộc rà soát literature khá triệt để, còn
trống. **[khối tuyên bố]** Đó chính là claim, được khoanh vùng chính xác:
hệ thống localization đầu tiên *xuất ra* một calibrated pose distribution
từ một Gaussian-splatting map, với evidential semantics được lan truyền
xuyên suốt end-to-end. Từng từ trong câu đó đều có "trọng lượng" riêng.

---

## Phần: USGLoc — 12 phút

### 15. Key idea *(~2 phút)*
Vacuity — cái mass "tôi
không biết" được sinh ra khi một observation được
encode, và sau đó *không bao giờ bị bỏ đi*. Được fuse theo từng object — kèm một conflict correction
Khi matching: nó làm gate cho các correspondence. 
Khi refinement: được blend vào thành weight theo từng pixel — đó chính là
rendering equation từ trước, giờ mang theo trust thay vì color. 
Tại optimum: một covariance. Ở output: một mixture. 

không có heuristic cho dynamic object, không có LLM quyết định cái gì đáng tin. Chính uncertainty của map
tự làm việc "gác cổng".

### 16. Contributions *(~1 phút)*

bản thân hệ thống localization theo calibrated distribution; matching được
gate theo vacuity, biến sự ambiguity trong retrieval thành một pose mixture
tường minh; refinement có weight theo vacuity kèm Laplace covariance; và —
có lẽ là cái quan trọng nhất mà ít ai để ý — một evaluation protocol lấy
calibration làm trung tâm, bởi vì minh không thể claim honesty mà không đo
lường nó.

### 17. At a glance *(~1.5 phút)*
Kiến trúc tổng thể. Xanh lá: cả query lẫn map đều
được encode vào cùng một evidential space — cùng một head, các vacuity có
thể so sánh được. Xanh dương: coarse stage — matching có gate, top-k
hypothesis kèm weight. Cam: fine stage — mỗi hypothesis được refine bằng
rendering, mỗi cái có covariance riêng. Vàng: mixture, được condense thành
một single Gaussian cho filter.

### 18. Problem formulation *(~1 phút)*
Chúng ta ước lượng posterior trên SE(3),
không phải argmax — với điều kiện phải calibrated, một property *có thể
kiểm chứng*. 
Cả hai phía đều mang cùng một bộ— geometry, appearance,
semantics, và vacuity. Điểm mấu chốt của sự đối xứng này: uncertainty là
first-class ở cả hai phía *trước khi* matching bắt đầu. Đó là
điều làm cho việc propagation trở nên khả thi ngay từ đầu.

### 19. Method 1a: shared evidential head *(~1 phút)*
Các công thức Dirichlet evidence, belief, vacuity. Hai
điều cần bổ sung. Cùng một head encode cả các observation khi build map và
các segment cua Fast-SAM ở thời điểm query — nên vacuity của map và vacuity
của query nằm trên cùng một scale; gate ở phần sau phụ thuộc vào điều đó.

### 20. Method 1b: primitives + "trả bài" cho Zadeh *(~2 phút)*
Xây dựng các object-level primitive: gom cụm các splat, lấy trung bình các
feature có weight theo confidence, fuse các opinion bằng Dempster's rule.

cliffhanger is solved. Hãy
nhớ câu hỏi: điều gì xảy ra khi các opinion confidently disagree? 

 với các object thực sự bị move: một nửa số view nói
là chair, nửa còn lại nói là box... và Dempster's rule fuse chúng thành
vacuity *thấp*. Object đã move lại trông *chắc chắn hơn* so với chính các
member của nó. 

Conflict — cái eta ở mẫu số — bị renormalize mất đi. Cách
sửa chỉ gọn trong một dòng: track conflict, cộng nó trở lại — effective
vacuity, bằng min của một và u cộng eta-bar. Tương đương với Yager's rule.
Một paradox năm mươi năm tuổi, một cách sửa vỏn vẹn một dòng, và nếu thiếu
nó thì toàn bộ weighting ở các bước sau sẽ chỉ sai hướng. Số liệu cụ thể ở
slide backup.

### 21. Method 2: gated matching *(~1.5 phút)*
Descriptor: CLIP feature cộng với geometry qua các MLP nhỏ, cùng một space
cho cả 2D và 3D. 
Score: cosine similarity nhân với evidential gate — e mũ
trừ lambda nhân với *cả hai* vacuity. Các cặp confident-confident sẽ
thống trị
Scene-level retrieval kế thừa gate này qua attention pooling,
và ambiguity hiện ra dưới dạng nhiều submap có score tương đương nhau —
mà chúng tôi *giữ lại*, dưới dạng một mixture, thay vì giả vờ rằng cái tốt
nhất là chắc chắn.

### 22. Method 3: refinement *(~1.5 phút)*
Vacuity được alpha-blend y hệt như color — cùng một kernel, thêm một channel, gần như miễn phí. Giờ mỗi pixel
được render biết map tin tưởng nó đến mức nào, và photometric error được
weight bằng e mũ trừ gamma U. 

Các đối tượng di chuyển, các vùng có kết cấu kém và ngữ nghĩa không chắc chắn được giảm trọng số liên tục 

Các weight phải được *detach*
khỏi computation graph ở mỗi iteration. Nếu gradient di qua chúng,
optimizer sẽ phát hiện ra rằng nó có thể giấu misalignment trong các vùng
vacuity cao thay vì sửa pose. 


### 23. Method 4: Laplace + temperature *(~1.5 phút)*
Từ optimum đến covariance: Laplace approximation — độ cong của loss. 
Đáy phẳng, covariance lớn; đáy sắc, covariance nhỏ. Và scale của residual có
nghĩa là các query bị degrade tự động nhận được over-covariance.

Nhan ra rang: các photometric residual thực tế có heavy tail và correlate với nhau, nên nó overconfident. 

Một temperature vô hướng duy nhất, được fit trên một held-out split bằng
median estimator, sửa lại scale. Trên synthetic benchmark của chúng tôi,
đó là ECE từ 0.50 xuống 0.16 — từ vô dụng thành honest — trong khi chất
lượng *ranking* vốn đã tốt từ đầu. Informative trước, honest sau. minh cần
cả hai.

### 24. Method 5: mixture *(~1 phút)*
Tổng hợp lại: weight từ retrieval nhân với render fidelity, prune những gì
không render lại được, và minh có một Gaussian mixture trên SE(3). Khi
scene ambiguous, mixture thể hiện rõ tính multi-modal — được giữ lại,
không bị average mất đi. 
Cho filter: được moment-match về một single
Gaussian. Mixture là câu trả lời honest; việc condense là câu trả lời
practical; minh có được cả hai.

### 25. Algorithm *(~30 giây)*
Toàn bộ hệ thống trên một slide để tham khảo — các bước một đến ba được
learn, bốn đến sáu training-free. Tôi sẽ để nó hiển thị một lúc. **[dừng
lại, uống nước]**

---

## Phần: Đánh giá — 7 phút

### 26. Datasets *(~1.5 phút)*
Các benchmark tiêu chuẩn cộng với một semi-dynamic protocol — object được
sắp xếp lại giữa lúc mapping và lúc query, cộng với các map bị degrade có
chủ đích,

không chỉ là accuracy mà còn là liệu covariance có
*tăng lên* đúng lúc cần hay không. 

7Scenes có hai ground-truth convention không tương thích
với nhau, và cùng một method có thể chênh nhau tới mười sáu điểm phần
trăm về accuracy giữa hai convention đó. Mọi công trình hiện đại đều báo
cáo theo SfM pseudo-ground-truth; việc trộn lẫn hai cột số liệu một cách
âm thầm là lỗi trích dẫn phổ biến nhất trong lĩnh vực này. Hãy kiểm tra kỹ
phần chú thích nhỏ của các baseline minh so sánh.

### 27. Metrics *(~2 phút)*
Accuracy như thường lệ. Rồi đến phần trọng tâm: bốn metric về calibration,
mỗi cái trả lời một câu hỏi. Coverage: covariance có honest không — các
vùng chi-square có "bắt" được sự thật đúng tỷ lệ đã công bố không? ECE:
cũng vậy, nhưng gói gọn trong một con số. AUSE: uncertainty
có informative không — nếu tôi loại bỏ những query mà model *nói* là
uncertain, error có giảm giống như oracle không? NLL: toàn bộ
distribution, bao gồm cả tính multi-modal. 
Và lý do minh
cần cả hai nhóm metric: một covariance có thể rank error hoàn hảo trong
khi vẫn nhỏ hơn thực tế mười lần, và ngược lại. Honest *và* informative —
đó là hai kỳ thi riêng biệt.

### 28. Vị trí hiện tại của lĩnh vực *(~1.5 phút)*
Bậc thang accuracy trên 7Scenes, tất cả như đã công bố. Xuống tới
SplatHLoc ở nửa centimet. Giờ nhìn vào cột bên phải **[rê ngón tay xuống]**:
none, none, none, none, internal-only, none. Mục tiêu của chúng tôi
*không phải* là thắng cột này bằng một phần mười milimet — mà là đứng
trong nhóm đó trong khi là hàng duy nhất có gì đó khác "none" ở cột bên
phải. Một cuộc đấu công bằng về accuracy, không có đối thủ về honesty.

### 29. Tình trạng triển khai & bằng chứng trên synthetic data *(~1.5 phút)*
Chúng tôi đang ở đâu: toàn bộ pipeline đã được implement và có unit test;
các con số ở đây đến từ một synthetic benchmark được kiểm soát — kết quả
trên real data là công việc đang tiến hành, nên hãy xem đây là bước kiểm
tra mechanism, được ghi nhãn một cách trung thực. Ba điều nó đã cho thấy.
Map phân tách nội dung đã move khỏi nội dung static chỉ dựa vào vacuity —
0.67 so với 0.16, không cần threshold. Mixture chứng minh giá trị của nó
— ép nó về một hypothesis duy nhất thì accuracy giảm một phần ba. Và câu
chuyện về temperature **[bảng số liệu]**: raw Laplace rank tốt nhưng nói
dối; sau khi temper thì làm được cả hai; một fixed covariance — điều mà
một filter giả định ngày nay — không rank được gì cả.

---

## Phần: Tầm nhìn tương lai — 3 phút

### 30. Open problems *(~2 phút)*
Nơi mọi thứ mở rộng ra ngoài một hệ thống duy nhất. Calibration dưới sự
drift — một temperature được fit hôm nay có thể sai trong một tháng nữa;
các map tự-recalibrate suốt đời vẫn còn bỏ ngỏ. Các foundation model cũng
mắc cùng căn bệnh — confidence của VLM cũng chưa được calibrate; cùng một
liều thuốc có thể áp dụng được, chỉ là bệnh nhân lớn hơn. Uncertainty như
một control signal — vacuity theo nghĩa đen cho robot biết nên nhìn vào
đâu tiếp theo mới có ích; đó là active perception xây trên nền các
evidential map. Guarantee — các phương pháp conformal hứa hẹn coverage
theo cấu trúc toán học; chưa ai làm điều đó cho pose posterior ở quy mô
cả scene. Và đóng vòng lặp với filter và các chuyến bay thực tế

### 31. Tổng kết *(~1 phút)*
Bốn câu. Khớp output với người tiêu thụ nó — filter cần distribution,
không cần point. Cho map một channel "tôi không biết" tường minh và
*dùng nó ở mọi nơi*. Làm cho uncertainty sống sót đến tận output — và
temper nó, vì raw Laplace luôn tự "tâng bốc" bản thân. Và tôn trọng những
chi tiết kỹ thuật nhỏ — Dempster giấu conflict, gradient có thể "lách
luật" qua các weight; sự khác biệt giữa toán học đẹp đẽ và một hệ thống
thực sự chạy được nằm chính xác ở hai footnote này. Cảm ơn mọi người —
rất hoan nghênh câu hỏi.

---

## Các slide backup — khi nào nên dùng đến

- **Dempster bằng số liệu cụ thể** — nếu có ai chất vấn câu chuyện về Zadeh
  hoặc hỏi "sao không dùng Yager luôn cho mọi trường hợp?" (Trả lời: với
  các source *đồng thuận*, việc reinforcement của Dempster là điều mong
  muốn; chúng tôi giữ lại nó và track conflict riêng — dạng "effective
  vacuity" giữ được cả hai hành vi.)
- **Việc detach gradient khỏi weight (IRLS)** — nếu có ai làm về
  optimization hỏi về việc lấy đạo hàm qua các weight.
- **Các training objective** — nếu có ai hỏi cái gì được learn và cái gì
  training-free.

## Hỏi & Đáp dự kiến

1. **"Sao không dùng luôn particle filter — minh sẽ có ngay một distribution
   miễn phí."** Particle cho ra một belief kiểu empirical, chưa calibrated
   và tốn kém — hàng trăm lần render cho mỗi query; và các hệ thống hiện
   có dù sao cũng chỉ báo cáo MAP. Chúng tôi lấy mixture từ k hypothesis
   đã refine cộng với các component Laplace dạng analytic, rồi *verify*
   calibration. Không có gì ngăn một particle back-end áp dụng cách
   weighting vacuity của chúng tôi — đó là một hướng mở rộng tốt.
2. **"USGLoc khác UGS-Loc cụ thể ở chỗ nào?"** Ba trục khác biệt: *nguồn
   gốc* của uncertainty (Fisher information của họ mang tính geometric,
   post-hoc, từ một fixed map; vacuity của chúng tôi mang tính evidential
   và semantic, được xây dựng ngay tại thời điểm mapping), *mục đích sử
   dụng* (họ dùng để guide search nội bộ; chúng tôi còn dùng để gate
   matching, weight pixel, và shape output), và *cái gì được xuất ra*
   (particle tốt nhất của họ; calibrated GMM của chúng tôi). Chúng tôi
   cũng cite họ như baseline mạnh nhất về accuracy.
3. **"Một temperature vô hướng duy nhất chẳng phải quá thô sao?"** Đó là
   mức tối thiểu trung thực nhất: một parameter duy nhất, fit trên
   held-out data, mang lại lợi ích calibration rất lớn. Các covariance
   head theo từng DoF hoặc được learn đều nằm trong hướng phát triển
   tương lai — nhưng mỗi parameter thêm vào là một thứ nữa có thể âm thầm
   overfit trên chính calibration split.
4. **"Vacuity có bắt được geometric ambiguity — như textureless wall
   không?"** Có, một phần và gián tiếp — thông qua scale của residual
   trong bước Laplace, vì sự "phẳng" về photometric làm phóng đại
   covariance. Semantic vacuity nhắm vào content trust. Hai cái này bổ
   sung cho nhau; việc fuse một geometric term tường minh (kiểu Fisher
   information) với evidential term là một hướng mở rộng còn bỏ ngỏ.
5. **"Vì sao filter nên tin covariance của minh hơn là một hằng số đã được
   tune sẵn?"** Đó chính xác là điều mà fixed-covariance baseline kiểm
   tra: cùng những pose đó, fixed covariance tốt nhất có thể — AUSE của
   nó tệ hơn khoảng 10 lần, nghĩa là nó không thể phân biệt được query
   tốt với query xấu. Một hằng số đã tune thì calibrated *trung bình*
   nhưng lại sai ở từng query riêng lẻ.
6. **"7Scenes đã saturate rồi — sao vẫn evaluate ở đó?"** Để có thể so
   sánh được về accuracy (ai cũng báo cáo trên đó) — các cột về
   *calibration* mới là nơi contribution thực sự nằm ở, và chúng hoàn
   toàn mới trên bất kỳ benchmark nào. Semi-dynamic protocol là bãi thử
   nghiệm mang tính phân biệt thực sự.
7. **"Kết quả trên real data thì sao?"** Đang được thực hiện; synthetic
   benchmark đã xác thực được mechanism (vacuity của dynamic object, hành
   vi của mixture, việc temper). Bộ công cụ cho 7Scenes và map-bridge
   format đã có sẵn trong code đã công bố.

## Danh sách cắt giảm cho buổi 25 phút

Bỏ, theo thứ tự: slide 4 (nhập môn localization — gộp một câu vào phần mở
đầu), slide 8 (nhập môn DST — phần "trả bài" Zadeh ở Method 1b tự nó đủ
mang được ý), slide 11–13 (các bảng related work — chỉ giữ lại slide 14,
ô trống trong design space), slide 18 (problem formulation), slide 25
(algorithm), slide 30 (open problems — gộp lời mời vào phần tổng kết).
Kết quả: khoảng 24 phút.
