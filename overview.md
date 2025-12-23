# AML Project – High-Level Substeps (Story → Checklist)

## 0. Big Picture

Goal: decide whether a cooking procedure is done correctly by learning from video features, not raw video.

---

## 1. Understand the Data (CaptainCook4D)

- Each recipe = full cooking video
- Video is split into **steps**
- Each step has:
  - video segments
  - correctness label (correct / incorrect)
  - error type (optional analysis)
- Each recipe also has a **task graph** (valid step orders)

---

## 2. Use Pre-extracted Video Features

- Do **not** train video models from scratch
- Load provided embeddings from:
  - **Omnivore**
  - **SlowFast**
- Each short video sub-segment → feature vector

Relationship:

> Video clip → Omnivore / SlowFast → numerical embedding

---

## 3. Step-Level Mistake Detection (Core Requirement)

### 3.1 Prepare Inputs

- For each recipe step:
  - collect its sub-segment feature vectors
  - form a sequence or aggregated representation

### 3.2 Train Baseline Models

- **V1 (MLP)**
  - aggregate features
  - binary classification: correct vs incorrect
- **V2 (Transformer)**
  - attend over sub-segment feature sequence
  - binary classification

### 3.3 Evaluate

- Metrics: Accuracy, Precision, Recall, F1, AUC
- Extra: analyze performance per error type

---

## 4. Improve / Extend Baselines (Still Step-Level)

- Propose a new simple model (e.g. RNN / LSTM)
- Compare against V1 and V2
- (Optional) extract features with a new backbone

---

## 5. Extension: From Steps to Full Recipe Verification

### 5.1 Step Localization

- Run a step-localization model on full videos
- Output: (start_time, end_time) per detected step
- Average features inside each interval → step embedding

---

### 5.2 Task Graph Matching

- Encode task-graph step descriptions (text)
- Match detected steps ↔ task-graph nodes
- Use aligned video–text embeddings
- Result: “realized” task graph from the video

---

### 5.3 Whole-Recipe Classification

- Input:
  - sequence of detected steps or
  - updated task graph
- Model:
  - Transformer (sequence) or
  - GNN (graph)
- Output:
  - recipe-level label: correct / incorrect

---

## 6. Final Output

- Write an 8-page report (CVPR style)
- Explain:
  - baselines
  - extension
  - experiments
  - insights and limitations
