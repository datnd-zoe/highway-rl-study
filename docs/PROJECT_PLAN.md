# 📘 highway-rl-study: Kế hoạch Nghiên cứu & Nhật ký Tiến độ (Project Plan & Progress Notes)

---

## 1. Bối cảnh & Mục tiêu Nghiên cứu (Context & Objectives)

* **Tên dự án**: `highway-rl-study`
* **Câu hỏi nghiên cứu (Research Question)**:
  > *"Trọng số phạt va chạm (`collision_reward`) trong hàm thưởng ảnh hưởng thế nào đến tỉ lệ va chạm (`collision_rate`) và tốc độ trung bình (`mean_speed`) của agent lái xe tự hành trong môi trường `highway-env`?"*
  >
* **Mục đích**:
  * Làm bằng chứng năng lực cho hồ sơ thực tập **R&D Physical AI / Robotics / RL Engineer**.
  * Thể hiện tư duy nghiên cứu thực nghiệm chuẩn mực: có giả thuyết, cô lập biến độc lập, kiểm soát biến phụ thuộc, so sánh baseline, lặp lại qua nhiều seeds (mean ± std), viết test tự động (`pytest`), và logging minh bạch (không bịa số liệu).

---

## 2. Ràng buộc Kỹ thuật & Công nghệ (Technical Stack)

* **Hệ điều hành**: Windows 11, PowerShell.
* **Môi trường Python**: Python 3.12.3 trong môi trường ảo `.venv/`.
* **Thư viện cốt lõi**:
  * `gymnasium` (v1.4.0)
  * `highway-env` (v1.12.1)
  * `stable-baselines3` (v2.9.2)
  * `torch` (v2.15.0 - chạy trên CPU)
  * `numpy` (v2.5.5), `matplotlib` (v3.11.2), `pyyaml`, `pandas`, `pytest` (v9.2.0).
* **Quy chuẩn mã nguồn**:
  * Chạy mượt mà trên **laptop CPU** (không đòi hỏi GPU).
  * Code ngắn gọn, có Type Hints và Docstrings tiếng Anh chuẩn PEP 8.
  * Comment giải thích bằng tiếng Việt ở các khối logic/thiết kế thuật toán quan trọng.

---

## 3. Phát hiện Quan trọng & Quyết định Thiết kế Reward (Key Discovery)

Qua việc đọc trực tiếp mã nguồn `highway-env` v1.12.1 (`highway_env/envs/highway_env.py`):

### 3.1. Vấn đề của `normalize_reward = True`

* Mặc định của môi trường là `normalize_reward: True`, dùng phép chiếu tuyến tính `lmap`:
  $$
  \text{reward\_norm} = \frac{v - \text{collision\_reward}}{(\text{high\_speed\_reward} + \text{right\_lane\_reward}) - \text{collision\_reward}}
  $$
* Khi thay đổi `collision_reward` từ `-1` sang `-20`:
  * Với mức phạt `-20`, toàn bộ dải reward của các trạng thái bình thường (không va chạm) bị nén chặt vào khoảng hẹp `[0.976, 1.0]`.
  * Hậu quả: Agent bị mất tín hiệu gradient phân biệt giữa lái tốt và lái trung bình, làm méo mó bản chất của việc so sánh.

### 3.2. Quyết định khoa học (Đã chốt)

1. **Tắt chuẩn hóa**: Đặt `normalize_reward: false` cho toàn bộ các cấu hình PPO (`ppo_low`, `ppo_mid`, `ppo_high`).
2. **Cô lập biến độc lập**:
   * Giữ nguyên: `high_speed_reward = 0.4`, `right_lane_reward = 0.1`, `lane_change_reward = 0.0`.
   * Chỉ thay đổi duy nhất: `collision_reward` (`-1.0`, `-5.0`, `-20.0`).
3. **Quy tắc so sánh metric**:
   * **`mean_reward`**: Chỉ có ý nghĩa so sánh nội bộ trong cùng 1 config (ví dụ qua các checkpoint/seeds của chính nó). **Không so sánh chéo `mean_reward` giữa các config khác nhau** vì thang điểm tối thiểu khác nhau.
   * **`collision_rate`** và **`mean_speed`**: Là 2 metric khách quan chính dùng để so sánh trực tiếp giữa các config và baseline.

---

## 4. Thiết kế Thí nghiệm Chi tiết (Experimental Design)

### 4.1. Cấu hình biến độc lập

| Config            | Trọng số `collision_reward` | Mục đích / Giả thuyết                                                                |
| ----------------- | ------------------------------- | ----------------------------------------------------------------------------------------- |
| `ppo_low.yaml`  | `-1.0`                        | Phạt nhẹ: Agent ưu tiên đi nhanh, chấp nhận rủi ro va chạm cao                   |
| `ppo_mid.yaml`  | `-5.0`                        | Phạt vừa (Benchmark tham chiếu chính)                                                 |
| `ppo_high.yaml` | `-20.0`                       | Phạt nặng: Agent lái xe thận trọng cao, có thể chấp nhận đi chậm để an toàn |

### 4.2. Huấn luyện (Training)

* **Số bước (Timesteps)**: 100,000 bước / run.
* **Seeds huấn luyện**: `[0, 1, 2]` (3 runs độc lập cho mỗi cấu hình để tính trung bình và độ lệch chuẩn $\text{mean} \pm \text{std}$).
* **Kế hoạch chạy**:
  1. Chạy thử nghiệm smoke test: 2,000 bước để kiểm tra pipeline.
  2. Chạy 1 run mẫu đầy đủ (`ppo_mid`, seed 0) để đo thời gian thực tế trên CPU và xem learning curve trước khi chạy hàng loạt.

### 4.3. Đánh giá (Evaluation)

* **Tập seed đánh giá**: Cố định **50 episodes** với seed từ `100` đến `149` (tách biệt hoàn toàn với seed huấn luyện để tránh data leakage / overfitting).
* **Định nghĩa Metric**:
  * `collision_rate` = $\frac{\sum \text{episodes có va chạm}}{\text{Tổng số episodes (50)}}$.
  * `mean_speed` = Vận tốc dọc trung bình qua từng step trong episode, sau đó lấy trung bình qua 50 episodes.
  * `mean_episode_length` = Số bước sống sót trung bình trước khi va chạm hoặc hết thời gian (duration = 40s).
  * `mean_reward` = Tổng tích lũy reward trung bình mỗi episode (chỉ so sánh nội bộ).
* **Tổng hợp kết quả**: Tính $\text{mean} \pm \text{std}$ qua 3 seed huấn luyện (mỗi seed đã được evaluate qua 50 episodes cố định).

### 4.4. Baselines

1. **Random Agent**: Chọn ngẫu nhiên đều 5 actions (`LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`) làm cận dưới (lower bound).
2. **Rule-based "Defensive Driver"**:
   * Sử dụng thông tin xe trực tiếp từ simulator (`env.road`, `env.vehicle`).
   * Quy tắc:
     * Mặc định chọn `FASTER`.
     * Nếu có xe phía trước trong làn hiện tại cách $< 40\text{m}$:
       * Kiểm tra làn trái (nếu trống $> 40\text{m}$ phía trước và an toàn phía sau) $\rightarrow$ `LANE_LEFT`.
       * Nếu làn trái không được, kiểm tra làn phải $\rightarrow$ `LANE_RIGHT`.
       * Nếu cả 2 làn đều kẹt $\rightarrow$ `SLOWER` để giảm tốc giữ khoảng cách an toàn.
     * Nếu không bị chặn và không ở làn phải cùng $\rightarrow$ ưu tiên về làn phải (`LANE_RIGHT`).
   * Chạy đánh giá trên `baseline.yaml` (dùng cùng env spec với `ppo_mid`). Có bài test `pytest` xác minh agent này không đổi hành vi khi `collision_reward` đổi.

---

## 5. Nhật ký Tiến độ Thực hiện (Progress Notes)

### 🟢 Giai đoạn đã hoàn thành

* [X] **Bước 0: Môi trường & Dependencies**

  * Khởi tạo virtual environment `.venv/` với Python 3.12.3.
  * Cài đặt thành công: `gymnasium`, `highway-env`, `stable-baselines3`, `torch`, `numpy`, `matplotlib`, `pyyaml`, `pytest`, `pandas`.
  * Hoàn thành phân tích mã nguồn `highway-env` v1.12.1 về cơ chế `normalize_reward`.
* [X] **Bước 1: Khởi tạo khung dự án (Scaffold & Git)**

  * Tạo cấu trúc cây thư mục chuẩn: `src/`, `configs/`, `tests/`, `results/`, `models/`.
  * Viết `.gitignore` (bỏ qua `.venv`, checkpoint model `*.zip`, video/gif nặng).
  * Tạo `requirements.txt`.
  * Git commit lần 1: `6d6133d` (*"chore: init repo scaffold with directories, .gitignore, requirements.txt"*).
  * Gắn kết nối remote GitHub: `https://github.com/datnd-zoe/highway-rl-study.git`.
* [X] **Bước 2: Cấu hình Thực nghiệm (`configs/`)**

  * Tạo 4 file cấu hình YAML hoàn chỉnh:
    * `configs/baseline.yaml`
    * `configs/ppo_low.yaml`
    * `configs/ppo_mid.yaml`
    * `configs/ppo_high.yaml`
  * Viết script test parse YAML đảm bảo cấu trúc dữ liệu đồng nhất.
  * Git commit lần 2: `0e4bff6` (*"feat: add experiment configs (baseline, ppo_low, ppo_mid, ppo_high)"*).

---

### 🟡 Các bước tiếp theo cần triển khai (Next Steps)

* [X] **Bước 3: `src/env_factory.py`**
  * Viết hàm `make_env(config_path, render_mode=None)` nhận config YAML và khởi tạo môi trường `highway-env` chuẩn hóa.
  * Đảm bảo ghi đè chính xác `collision_reward` và ép `normalize_reward: False`.
* [X] **Bước 4: `src/baseline.py`**
  * Xây dựng lớp `RandomAgent` và `DefensiveDriverAgent`.
* [ ] **Bước 5: `src/train.py`**
  * Huấn luyện PPO qua Stable-Baselines3 với tham số từ YAML và seed xác định.
  * Lưu model checkpoint vào `models/`.
* [ ] **Bước 6: `src/evaluate.py`**
  * Vòng lặp đánh giá 50 episodes cố định (seed `100+i`).
  * Trích xuất các metric: `collision_rate`, `mean_speed`, `mean_episode_length`, `mean_reward`.
  * Lưu kết quả ra file CSV trong `results/`.
* [ ] **Bước 7: `src/plot_results.py`**
  * Đọc kết quả từ CSV, vẽ đồ thị so sánh $\text{mean} \pm \text{std}$ (bar chart / boxplot).
* [ ] **Bước 8: `tests/`**
  * Viết test tự động với `pytest`:
    * Khởi tạo env thành công từ config.
    * Đảm bảo seed cố định cho kết quả mô phỏng lặp lại 100%.
    * Output format của `evaluate.py` khớp schema CSV.
    * Baseline agents cho cùng metric bất kể `collision_reward`.
* [ ] **Bước 9: Tài liệu báo cáo (`README.md` & `EXPERIMENTS.md`)**
  * `README.md` (English): Research question, architecture, setup, run instructions, Limitations, và Results placeholder (TODO).
  * `EXPERIMENTS.md`: Nhật ký thực nghiệm dạng bảng (ngày, config, seed, số bước, metrics, nhận xét).
* [ ] **Bước 10: Smoke Test toàn bộ pipeline**
  * Chạy thử 2,000 bước từ train $\rightarrow$ eval $\rightarrow$ plot để xác nhận pipeline không gặp lỗi runtime.
