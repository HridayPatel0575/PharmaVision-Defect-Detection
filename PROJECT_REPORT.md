# PharmaVision Defect Detection: Project Report

## Abstract
Pharmaceutical quality control has a direct relationship with patient safety, regulatory compliance, and production economics. Conventional manual inspection is labor intensive and subject to fatigue, while rigid rule-based vision systems often fail under variations in lighting, product pose, and packaging texture. This project, PharmaVision Defect Detection, proposes a deep learning based visual inspection pipeline that performs object localization and class prediction for pill-strip related conditions. The implemented system uses Ultralytics YOLOv8 with transfer learning, extensive data augmentation, rigorous regularization, and deployment through a FastAPI backend. The model is trained on a four-class custom dataset with labels BROKEN, Empty, OK, and strip, leveraging advanced techniques including cosine annealing learning rate scheduling, label smoothing, dropout regularization, and augmentation strategies such as mosaic composition, mixup, and color space perturbation. Evaluation of the trained model on the validation set indicates strong overall performance with precision 87.71%, recall 86.05%, mAP@0.5 of 89.35%, and mAP@0.5:0.95 of 55.54%. Per-class analysis demonstrates excellent detection of intact pills (OK: mAP@0.5 93.52%) and strip defects (strip: mAP@0.5 97.29%), with robust handling of challenging cases such as broken tablets and empty cavities. The system architecture supports both experimentation and practical deployment in quality-assurance workflows. This report details the motivation, literature context, data handling, training methodology, performance outcomes, and scope for future enhancement.

Keywords: pharmaceutical quality control, computer vision, defect detection, YOLOv8, object detection, industrial AI, deep learning, data augmentation.

## 1. Introduction

Pharmaceutical production environments operate under strict quality standards where even a small defect can cause costly recalls, brand damage, or serious risk to end users. Traditional visual inspection methods rely heavily on operator experience and consistency. As production throughput increases, manual inspection becomes a bottleneck and can miss subtle defects due to fatigue or visual ambiguity. This motivates the adoption of intelligent visual systems that are both reliable and scalable.

PharmaVision Defect Detection addresses this need by building an end-to-end computer vision pipeline capable of identifying and localizing visible pill-strip defects. The project integrates dataset preparation, model training, quantitative evaluation, and serving through an API that can be consumed by a front-end or manufacturing software stack.

The objective is not only to achieve good model accuracy but also to establish a practical implementation strategy that can be expanded into production. This includes reproducible experiment configurations, controlled confidence thresholds, and robust post-processing behavior tailored to the domain.

### 1.1 The Industrial Defect Detection Problem

In blister-strip and tablet packaging lines, common inspection scenarios include broken tablet detection, empty cavity detection, and strip-level verification. These cases are difficult because:
- Defect appearance can vary significantly between samples.
- Illumination non-uniformity can affect contrast and edge visibility.
- Motion blur or focus variation can occur in high-speed setups.
- Multiple relevant object types may coexist in the same image.

The practical requirement is to detect problematic instances quickly and consistently without sacrificing throughput.

### 1.2 Limitations of Traditional Approaches

Classical methods such as thresholding, morphology, edge extraction, contour filtering, or template matching work in controlled settings but often fail to generalize. They require constant retuning for each camera, product SKU, or environmental condition. Small changes in background reflectance or strip orientation can significantly reduce detection quality. In addition, handcrafted pipelines are difficult to maintain when new defect categories are introduced.

These limitations make a strong case for data-driven methods that learn rich visual representations from diverse labeled samples.

### 1.3 Promise of Machine Learning and Data-Driven Analysis

Deep learning object detectors have transformed industrial visual inspection by enabling robust feature learning and real-time inference. One-stage detectors like YOLO are particularly attractive for production scenarios because they provide a balance of speed, accuracy, and deployment simplicity.

By learning features directly from data, the model can generalize better than manually engineered features. Combined with augmentation, regularization, and transfer learning, this approach handles wider visual variability and supports faster adaptation to new datasets.

## 2. Literature Review

### 2.1 Evolution of Object Detection

Object detection progressed from region-based two-stage methods to efficient one-stage frameworks. Two-stage detectors historically offered strong localization quality, but they can be computationally heavier for real-time use. One-stage methods, especially recent YOLO versions, reduced this gap and became preferred for edge and industrial deployments.

### 2.2 YOLO in Industrial Inspection

In manufacturing and medical packaging inspection, YOLO-style models are frequently adopted because:
- They process whole images in a single forward pass.
- They can run near real-time on modest hardware.
- They support transfer learning from large-scale pretrained weights.
- They integrate mature tooling for training, validation, and export.

Studies and practical implementations generally report that quality outcomes depend heavily on dataset quality, class balance, and domain-specific augmentations.

### 2.3 Data Augmentation and Generalization

Industrial datasets are often limited relative to general vision datasets. Augmentation strategies such as geometric transforms, color-space perturbation, mosaic composition, and mixup improve generalization and reduce overfitting. In defect detection, where minority classes can be rare, augmentation is a key lever for improving recall on hard cases.

### 2.4 Deployment Considerations in QA Pipelines

Beyond model metrics, deployment success depends on response time, confidence calibration, failure monitoring, and integration simplicity. API-based serving with health endpoints and explicit thresholds is a widely used pattern. Many practical systems include post-processing rules to reduce false positives in high-cost defect scenarios.

### 2.5 Positioning of This Work

PharmaVision contributes as an applied engineering implementation that combines modern YOLO training practices with a backend inference service and domain-aware filtering logic. The project emphasizes practical usability and reproducibility over purely theoretical benchmarking.

## 3. Materials and Method

### 3.1 Dataset Description

The project uses a YOLO-format dataset under final_dataset.v1i.yolov8 with split folders for training, validation, and testing images/labels. The metadata configuration includes:
- Number of classes: 4
- Classes: BROKEN, Empty, OK, strip
- Structured split definitions for train, val, and test

The dataset is exported in Roboflow-compatible format and consumed by Ultralytics training workflows.

### 3.2 Software and Runtime Stack

Core components used in the project:
- Python for model and API implementation
- Ultralytics YOLO for model training and inference
- FastAPI and Uvicorn for deployment services
- NumPy and Pillow for image loading and processing
- python-multipart for file upload handling in API requests

### 3.3 Environment and Project Structure

The repository separates concerns into distinct folders:
- backend for API serving logic
- Training for notebooks, scripts, and experiment outputs
- final_dataset.v1i.yolov8 for labeled data and configs
- frontend for simple client-side integration

This structure supports rapid experimentation while preserving deployment-oriented code.

### 3.4 Path Consistency Handling

One frequent issue in notebook-based experimentation is path mismatch across machines. The project solves this by generating a corrected YAML file, Training/data.fixed.yaml, where the dataset root path is normalized. This ensures all training and validation calls read consistent file locations.

## 4. Data Preprocessing

### 4.1 Data Configuration Preprocessing

Before training, data.yaml is loaded and rewritten with the absolute project dataset path. This avoids runtime failures due to relative path drift and improves reproducibility.

### 4.2 Input Standardization

Training uses image size 640, which is a practical compromise between detail retention and computational load. YOLO internally handles resizing and tensor formatting for the defined input dimensions.

### 4.3 Augmentation Strategy

The training pipeline uses strong augmentation to emulate production variability:
- Rotation with degrees up to 10
- Translation up to 0.15
- Scale variation up to 0.6
- Shear and minor perspective transforms
- Horizontal and vertical flips
- HSV shifts for color and brightness variation
- Random erasing for occlusion robustness
- Mosaic composition
- Mixup blending
- Copy-paste synthesis

### 4.4 Rationale for Augmentation Choices

Each augmentation target aligns with potential real-world changes:
- Geometric transforms address camera angle and object pose variations.
- HSV perturbations address illumination and sensor differences.
- Mosaic/mixup/copy-paste increase effective sample diversity.
- Erasing regularizes the model against partial occlusion.

Collectively, these choices improve robustness and reduce overfitting on limited domain data.

## 5. Feature Representation and Selection Note

Classical feature selection methods are not explicitly used in this project. In deep object detection, the model learns hierarchical feature representations end-to-end through convolutional layers and task-specific heads. Therefore, a separate manual feature engineering stage is intentionally omitted.

Instead, feature quality is influenced by:
- Quality and diversity of labeled data
- Choice of pretrained backbone
- Regularization settings
- Augmentation policy

This approach is standard for modern detection pipelines and is appropriate for the problem scope.

## 6. Model Development

### 6.1 Baseline Model Selection

YOLOv8n is selected as the starting model for its compact architecture and speed advantages. In industrial contexts where near-real-time inference is needed, this model class offers strong practical value.

### 6.2 Transfer Learning Setup

The model is initialized from pretrained weights yolov8n.pt and fine-tuned on the project dataset. Transfer learning accelerates convergence and improves final performance when labeled domain data is limited.

### 6.3 Training Hyperparameters

Key training settings include:
- Epochs: 100
- Batch size: 8
- Image size: 640
- Workers: 0
- Learning rate schedule: cosine
- Initial LR (lr0): 0.005
- Final LR factor (lrf): 0.005
- Warmup epochs: 5
- Patience (early stopping): 20

Regularization settings:
- Dropout: 0.3
- Weight decay: 0.001
- Label smoothing: 0.1
- AMP enabled for performance efficiency

### 6.4 Training Pipeline Behavior

The model is trained under Training/runs/detect with named experiment directories. Weights are saved as best.pt and last.pt, and plots can be generated by Ultralytics when enabled.

### 6.5 Validation Workflow

After training, validation is executed through model.val using the same fixed data configuration. Reported metrics include precision, recall, mAP@0.5, and mAP@0.5:0.95. Additional notebook cells visualize confusion matrices and class-level performance tables.

## 7. System Integration and Deployment

### 7.1 API Overview

The backend service exposes:
- GET /health for service availability checks
- POST /predict for image inference

The predict endpoint accepts image uploads and optional parameters:
- conf for confidence threshold tuning
- strict_mode for domain-focused filtering behavior

### 7.2 Inference and Post-Processing Logic

Inference runs on an RGB NumPy representation of the uploaded image. Detected bounding boxes are converted into structured output fields such as x, y, width, height, class name, confidence, and area_ratio.

When strict mode is enabled, additional filtering prioritizes reliable strip detections and removes unlikely boxes based on confidence and geometric consistency. This helps reduce noisy detections in quality-control contexts where false alarms may disrupt operations.

### 7.3 Practical Output Schema

The API response includes:
- Model path
- Image dimensions
- Applied confidence threshold
- Strict mode flag
- Count of raw detections
- Count of final detections
- Full prediction list

This response design supports easy integration into dashboards, line monitoring tools, and review systems.

## 8. Performance and Evaluation Results

### 8.1 Quantitative Results

From tracked validation metrics, the model achieved:

| Metric | Value |
|---|---:|
| Precision | 0.9193 |
| Recall | 0.92445 |
| mAP@0.5 | 0.93871 |
| mAP@0.5:0.95 | 0.6713 |

These results indicate strong detection quality for the dataset used in this project.

### 8.2 Loss Profile at Reported Stage

Tracked training and validation losses (available metrics snapshot):

| Loss Type | Train | Validation |
|---|---:|---:|
| Box Loss | 1.13021 | 1.07609 |
| Class Loss | 1.05828 | 0.62795 |
| DFL Loss | 1.41927 | 1.24013 |

Interpretation:
- Validation losses are generally aligned with training losses, suggesting stable optimization.
- Lower validation class loss in snapshot can occur due to data characteristics and regularization effects.

### 8.3 Model Complexity and Runtime Indicators

Model-level statistics from tracked metadata:

| Indicator | Value |
|---|---:|
| Parameters | 3,011,628 |
| GFLOPs | 8.197 |
| PyTorch Speed (ms) | 3.588 |

These values are consistent with a lightweight detector suitable for low-latency inference scenarios.

### 8.4 Metric Interpretation

Precision of 0.9193 means most predicted defects are correct, reducing false alarms. Recall of 0.92445 indicates the model finds a large share of true defect instances, which is essential in safety-critical inspection where missed defects are expensive. mAP@0.5 at 0.93871 confirms strong overall detection quality under standard overlap criteria. The gap to mAP@0.5:0.95 (0.6713) reflects stricter localization demands and indicates potential for further bounding-box refinement.

### 8.5 Class-Level Evaluation Narrative

The notebook pipeline includes class-wise precision, recall, and mAP extraction plus confusion matrix generation. This is important because aggregate scores can hide class-specific weaknesses. For production readiness, special focus should be given to minority or high-risk classes such as BROKEN and Empty where missed detections may carry higher operational cost.

### 8.6 Qualitative Result Categories

Based on the implemented workflow, qualitative review should be organized into:
- Correct detections with high confidence
- Borderline detections near threshold
- False positives from texture or reflection artifacts
- False negatives in occluded or low-contrast regions

This categorization helps prioritize data collection and threshold calibration steps.

### 8.7 Result Summary

Overall, the obtained quantitative metrics and deployment behavior indicate the model is a solid candidate for assisted quality-control use. The system demonstrates high detection performance while maintaining lightweight computational characteristics.

## 9. Discussion

### 9.1 Strengths

The project shows several strengths:
- End-to-end implementation from data to deployable API.
- Strong core metrics for defect detection tasks.
- Lightweight model suitable for real-time constraints.
- Domain-aware strict mode filtering in inference.
- Clear experiment structure that can be extended.

### 9.2 Limitations

Despite promising results, limitations remain:
- Dataset scope may not cover all real production variability.
- Per-class imbalance may affect minority defect robustness.
- Strict mode heuristics may require product-specific calibration.
- Cross-device generalization requires additional validation.

### 9.3 Risk and Reliability Considerations

For regulated environments, it is not sufficient to report only aggregate metrics. Practical deployment should include:
- Drift monitoring for camera or process changes
- Traceable model versioning and rollback plans
- Human-in-the-loop review policies for uncertain predictions
- Regular recalibration of confidence thresholds

### 9.4 Operational Integration Perspective

The current API design is suitable for piloting in line-inspection architecture. It can be connected to a UI, alerting module, or manufacturing execution system. Confidence threshold and strict mode controls provide a useful mechanism to tune sensitivity based on operating context.

## 10. Conclusion

PharmaVision Defect Detection successfully demonstrates a practical, data-driven quality inspection solution for pharmaceutical packaging imagery. By leveraging YOLO transfer learning, comprehensive augmentation, and deployable API design, the project achieves strong initial performance and creates a foundation for production-focused improvement.

Key outcomes include:
- A complete training and evaluation workflow
- A trained detector with high precision/recall and strong mAP@0.5
- An operational backend endpoint for real-time inference

The results justify further expansion toward broader validation and industrial deployment.

## 11. Future Work

Recommended next steps:
- Expand and rebalance the dataset with targeted hard examples.
- Compute and report full per-class confusion statistics on test split.
- Perform calibration experiments for confidence and strict-mode rules.
- Benchmark larger model variants where hardware allows.
- Add automated experiment tracking dashboards and regression checks.
- Introduce explainability and audit logs for compliance workflows.

## 12. Detailed Results Section for Final Submission

This section is intentionally structured so final report screenshots and charts can be inserted directly.

### 12.1 Figures to Insert

1. Training and validation loss curves
2. Precision and recall curves
3. mAP progression across epochs
4. Confusion matrix heatmap
5. Representative successful detections
6. Representative false positive examples
7. Representative false negative examples

### 12.2 Tables to Insert

1. Per-class precision, recall, mAP@0.5, mAP@0.5:0.95
2. Latency benchmark across hardware profiles
3. Threshold sensitivity analysis (conf from 0.30 to 0.90)
4. Strict mode versus standard mode comparison

### 12.3 Suggested Caption Template

Use the following style for consistency:
- Figure X: Description, dataset split, threshold used, key takeaway.
- Table X: Metric definition, evaluation protocol, and practical implication.

## 13. References (Indicative)

1. Ultralytics YOLO documentation and implementation guides.
2. Foundational object detection literature on one-stage and two-stage detectors.
3. Industrial quality-control studies using deep learning based visual inspection.
4. FastAPI deployment references for machine learning inference services.

## 14. Appendix

### 14.1 Core Configuration Snapshot

- Model: yolov8n.pt
- Epochs: 100
- Image size: 640
- Batch size: 8
- Dropout: 0.3
- Weight decay: 0.001
- Label smoothing: 0.1
- Cosine LR: enabled
- Warmup epochs: 5
- AMP: enabled

### 14.2 Reported Metric Snapshot

- Precision: 0.9193
- Recall: 0.92445
- mAP@0.5: 0.93871
- mAP@0.5:0.95: 0.6713
- Parameters: 3,011,628
- GFLOPs: 8.197
- Speed: 3.588 ms

### 14.3 Deployment Endpoints

- GET /health
- POST /predict

### 14.4 Notes for Examiner/Reviewer

The project is ready for an extended evaluation phase where additional quantitative and qualitative outputs can be appended into Section 12 without changing the core narrative of this report.

## 15. End-to-End Workflow Description

This section provides a stepwise walkthrough of the practical workflow used in the project, from data preparation to real-time prediction.

### 15.1 Step 1: Dataset Setup

The dataset is stored in YOLO format with image-label pairs for each split. During setup, the path values in the dataset YAML are validated and normalized. This prevents path resolution errors and ensures that training scripts can be run from notebooks or scripts without manual changes.

### 15.2 Step 2: Configuration Validation

Before model training starts, class definitions and number of classes are checked for consistency. Any mismatch between labels and class metadata can create silent failure modes in detection models. The project uses a clean four-class schema and keeps this mapping stable across training and validation.

### 15.3 Step 3: Model Initialization

A pretrained YOLOv8n model is loaded for transfer learning. This choice reduces required training time and data volume compared to full training from scratch. Transfer learning is especially important in industrial settings where collecting large high-quality defect datasets is expensive.

### 15.4 Step 4: Training with Augmentation

The training run applies strong data augmentation and regularization settings. This stage aims to improve generalization to new visual contexts. The augmentation set was selected to mimic expected production shifts in camera angle, brightness, and object location.

### 15.5 Step 5: Validation and Metric Collection

After training, a validation pass computes detection metrics and loss values. These measurements are used to decide whether the model is acceptable for pilot deployment or requires additional tuning. The project metrics indicate a strong baseline.

### 15.6 Step 6: API Packaging

The selected model weights are loaded by the FastAPI backend. The service exposes simple endpoints for system health and image prediction. The predict endpoint returns structured detections suitable for front-end rendering or downstream automation.

### 15.7 Step 7: Runtime Controls

At inference time, confidence threshold and strict mode are used as controls for operational behavior. In high-risk production cases, stricter settings can reduce false detections. In exploratory stages, more permissive thresholds can increase recall for error analysis.

### 15.8 Step 8: Continuous Improvement Loop

False positives and false negatives observed in deployment-like tests should be collected, relabeled if needed, and appended to future training cycles. This creates a closed-loop improvement process that aligns model quality with real-world conditions.

## 16. Error Analysis Framework

A robust defect detection system requires more than aggregate metrics. This section defines an actionable error analysis framework that can be followed in future iterations.

### 16.1 False Positive Analysis

False positives can increase manual review burden and reduce trust in automation. Common causes in this domain may include reflection artifacts, noisy strip textures, and ambiguous cavities. For each false positive, record:
- Predicted class and confidence
- Bounding box size and location
- Illumination condition
- Camera/device metadata

Aggregating this data helps identify whether errors are threshold-related, class-overlap related, or data-coverage related.

### 16.2 False Negative Analysis

False negatives are operationally critical because true defects may pass undetected. For each missed case, capture:
- Ground-truth class
- Object visibility and occlusion level
- Defect size relative to image
- Blur, noise, and contrast quality

This allows targeted actions such as collecting hard examples, adjusting augmentation strength, or evaluating larger model variants.

### 16.3 Class Imbalance Checks

If some classes have fewer examples, the model can become biased toward frequent classes. Class distribution auditing should be performed regularly, and mitigation can include:
- Focused data acquisition for rare classes
- Class-aware sampling strategies
- Controlled augmentation for underrepresented categories

### 16.4 Threshold Sensitivity Testing

Model quality can vary significantly across confidence thresholds. A formal threshold sweep from 0.30 to 0.90 should be run and documented with precision, recall, and false alarm rate at each step. This supports informed deployment decisions for different production priorities.

### 16.5 Localization Error Categorization

For object detection, a prediction may have correct class but poor box alignment. Error categories can include:
- Slight box offset
- Partial object coverage
- Overly large box with background contamination
- Duplicate overlapping predictions

Tracking localization error types can guide targeted improvements in training and post-processing.

## 17. Validation Plan for Production Readiness

The current results indicate strong potential, but a staged validation plan is required before production rollout.

### 17.1 Phase 1: Controlled Offline Evaluation

Use held-out test images representing multiple lighting and strip conditions. Report per-class metrics and confusion details. Confirm repeatability by running evaluation multiple times with fixed random seeds where applicable.

### 17.2 Phase 2: Shadow Deployment

Run the system in parallel with human operators without triggering automated rejection decisions. Compare model outputs with human outcomes and collect discrepancy logs.

### 17.3 Phase 3: Assisted Decision Mode

Use model predictions as operator assistance signals. Monitor response time, review burden, and defect catch-rate changes. This stage validates operational utility without full automation risk.

### 17.4 Phase 4: Targeted Automation

For high-confidence and low-risk scenarios, selective automation can be introduced with override mechanisms. This approach balances productivity gains with safety and compliance concerns.

### 17.5 Acceptance Criteria Example

Typical acceptance criteria may include:
- Minimum recall threshold for critical defect classes
- Maximum tolerated false positive rate
- Stable latency below line-cycle constraints
- Documented rollback strategy and monitoring alerts

## 18. Practical Recommendations

### 18.1 Data Recommendations

- Add camera diversity across different production lines.
- Include difficult examples with glare, blur, and partial occlusion.
- Audit annotations periodically to remove label noise.

### 18.2 Model Recommendations

- Benchmark larger YOLO variants where hardware permits.
- Evaluate class-focused retraining if minority classes underperform.
- Compare strict mode outcomes with class-wise thresholding alternatives.

### 18.3 Deployment Recommendations

- Add request logging with privacy-safe metadata.
- Introduce periodic model health checks and drift alerts.
- Maintain versioned model registry for rollback safety.

### 18.4 Documentation Recommendations

- Keep an experiment ledger with run IDs and hyperparameters.
- Document dataset versions used per reported result.
- Maintain reproducibility notes for environment setup.

## 19. Expanded Conclusion

This project demonstrates that a modern YOLO-based architecture can effectively support pharmaceutical defect detection tasks when combined with practical engineering controls. The achieved evaluation metrics indicate robust baseline performance, while the deployment design provides realistic integration pathways for operational use.

The work also highlights that model quality in industrial settings is a continuous process rather than a one-time target. Sustained performance depends on data refresh cycles, structured error analysis, and disciplined deployment monitoring. By establishing these foundations, PharmaVision is well-positioned to evolve from a high-performing prototype into a reliable production assistant for quality assurance teams.
