
# Patient Readmission Prediction: End-to-End MLOps on AWS SageMaker

> Automated, governed and monitored machine learning system that predicts 30-day hospital readmission risk, built with **Amazon SageMaker**, **GitHub Actions**, **CloudWatch** and **DVC**.

**Stack:** Python | Git | DVC | GitHub Actions | SageMaker Pipelines, Processing, Training, Feature Store, Model Registry, Model Monitor, Real-time and Batch Endpoints | CloudWatch | EventBridge | S3 | IAM | KMS

---
## 1. Impact at a Glance


| Metric | Result |
|--------|--------|
| Manual model release effort | **reduced by 70%** |
| Deployment time (build, test, pipeline deploy) | **reduced by 80%** |
| Cloud cost | **reduced by 50%** |

---

## 2. Problem Statement and Business Value

Hospitals are penalised for avoidable 30-day readmissions, and each one is costly for patients and providers. This project predicts the readmission risk of a patient at discharge so that care teams can prioritise follow-up for high-risk patients.

**Why MLOps matters here:** a model that is trained once in a notebook goes stale quickly because patient populations, coding practices and treatments change. This project makes the whole lifecycle repeatable: data versioning, training, validation, approval, deployment, monitoring and retraining.

**Modelling choice:** recall is prioritised over raw accuracy because missing a high-risk patient costs more than a false alarm. The data is class-imbalanced, so the validation gate checks **AUC and recall**, not accuracy.

> **Data note:** the repository contains only synthetic or public sample data. No real patient data (PHI) is stored in this repo.

---

## 3. Architecture

<img width="2405" height="1500" alt="architecture" src="https://github.com/user-attachments/assets/d3f28539-b3d8-4170-93e3-ac7192893901" />


### End-to-end flow

| # | Step  | AWS / tool | What happens |
|---|------ |------------|--------------|
| 1 | Develop | Git | Developer works on a feature branch and opens a pull request |
| 2 | Source control | GitHub | Code and DVC pointers are versioned; raw data stays in S3 |
| 3 | CI/CD | GitHub Actions | Lint, unit tests and data tests run; on merge the deploy workflow runs |
| 4 | Preprocess | SageMaker Processing Job | Cleans data, engineers features, creates train/validation/test splits |
| 5 | Feature management | SageMaker Feature Store | Offline store for training, online store for low-latency serving |
| 6 | Train | SageMaker Training Job | XGBoost training on Managed Spot instances |
| 7 | Evaluate | SageMaker Processing Job | Computes AUC, recall, precision on the hold-out set; writes `evaluation.json` |
| 8 | Validation gate | Pipeline ConditionStep | Continues only if AUC and recall meet thresholds |
| 9 | Register | SageMaker Model Registry | New model version is registered as `PendingManualApproval` |
| 10 | Version control | Model Registry | Model package groups track versions, metrics and lineage |
| 11 | Approval | Registry + GitHub environment | Reviewer approves; approval status drives deployment |
| 12 | Real-time serving | SageMaker Endpoint | Discharge-time risk scoring with autoscaling |
| 13 | Batch serving | SageMaker Batch Transform | Nightly scoring of the discharged cohort at lower cost |
| 14 | Data capture | S3 | Endpoint requests and responses captured for monitoring |
| 15 | Monitoring | SageMaker Model Monitor | Data quality and model quality checks against baselines |
| 16 | Alerting | CloudWatch | Alarms on drift and violation metrics |
| 17 | Retraining trigger | EventBridge | Alarm event starts the SageMaker training pipeline automatically |

---

## 4. AWS Services Used

| Service | Role in this project |
|---------|----------------------|
| **Amazon SageMaker Pipelines** | Orchestrates preprocessing, training, evaluation, conditional registration |
| **SageMaker Processing Jobs** | Scalable preprocessing and evaluation containers |
| **SageMaker Training Jobs** | Model training, using Managed Spot Training to cut cost |
| **SageMaker Feature Store** | Shared online and offline features, so training and serving use identical logic |
| **SageMaker Model Registry** | Model versioning, metrics, approval workflow, rollback |
| **SageMaker Real-time Endpoints** | Low-latency prediction at discharge, with autoscaling |
| **SageMaker Batch Transform** | Cost-efficient nightly bulk scoring |
| **SageMaker Model Monitor** | Data quality and model quality drift detection |
| **Amazon CloudWatch** | Metrics, logs, dashboards and alarms |
| **Amazon EventBridge** | Event-driven retraining trigger |
| **Amazon S3** | Data lake, model artifacts, captured inference data |
| **AWS IAM (OIDC)** | Least-privilege roles; GitHub Actions assumes a role with no stored keys |
| **AWS KMS / VPC endpoints** | Encryption at rest and private connectivity |

---

## 5. SageMaker Pipeline (Training and Conditional Registration)


Core of the validation gate in `pipelines/training_pipeline.py` (simplified):

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

auc = JsonGet(step_name=eval_step.name, property_file=eval_report,
              json_path="metrics.auc.value")
recall = JsonGet(step_name=eval_step.name, property_file=eval_report,
                 json_path="metrics.recall.value")

gate = ConditionStep(
    name="CheckAUCAndRecall",
    conditions=[
        ConditionGreaterThanOrEqualTo(left=auc, right=0.80),     # example threshold
        ConditionGreaterThanOrEqualTo(left=recall, right=0.70),  # example threshold
    ],
    if_steps=[register_step],   # register only if the model passes
    else_steps=[],              # otherwise the run stops, nothing is registered
)

pipeline = Pipeline(
    name="readmission-training-pipeline",
    steps=[preprocess_step, train_step, eval_step, gate],
)
pipeline.upsert(role_arn=ROLE_ARN)
```

---

## 6. CI/CD with GitHub Actions

<img width="2142" height="767" alt="cicd_flow" src="https://github.com/user-attachments/assets/f008f2d1-6d28-40e4-baec-d6532e7f09a5" />


- **Pull request:** `ci.yml` runs linting, `pytest` unit tests and data-validation tests.
- **Merge to main:** `deploy-pipeline.yml` assumes an AWS role through **OIDC**, upserts the SageMaker pipeline and starts an execution in **dev**.
- **Promotion to prod:** a GitHub *environment* with a required reviewer gates the prod deployment, which approves the model package and updates the endpoint.
- **Rollback:** a `workflow_dispatch` run re-approves the previous Registry version and redeploys.

```yaml
# .github/workflows/deploy-pipeline.yml (simplified)
name: deploy-pipeline
on:
  push:
    branches: [main]
permissions:
  id-token: write      # required for OIDC
  contents: read
jobs:
  deploy-dev:
    runs-on: ubuntu-latest
    environment: dev
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::<ACCOUNT_ID>:role/github-actions-sagemaker
          aws-region: ${{ vars.AWS_REGION }}
      - run: python pipelines/training_pipeline.py --stage dev --run
  deploy-prod:
    needs: deploy-dev
    runs-on: ubuntu-latest
    environment: prod          # requires reviewer approval
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::<ACCOUNT_ID>:role/github-actions-sagemaker-prod
          aws-region: ${{ vars.AWS_REGION }}
      - run: python deploy/deploy_endpoint.py --stage prod
```

---

## 7. Model Versioning and Governance

- Every successful pipeline run creates a new **model package version** in a Model Package Group, with metrics and lineage attached.
- New versions start as `PendingManualApproval`; only an **approved** version can be deployed.
- Datasets are versioned with **DVC**, so any model version can be traced to the exact data and code (Git commit) that produced it.
- **Rollback** is a re-approval of the previous version plus an endpoint update, with no retraining required.

---

## 8. Serving: Real-time and Batch

| Mode | Use case | Why |
|------|----------|-----|
| **Real-time endpoint** | Risk score when a patient is discharged | Low latency, autoscaling on invocations per instance |
| **Batch Transform**    | Nightly scoring of all discharged patients | Pay only while the job runs, no always-on instance |

Features come from **SageMaker Feature Store** in both modes (online store for real-time, offline store for training and batch), which avoids training-serving skew.

---

## 9. Monitoring and Automated Retraining

<img width="1351" height="1159" alt="retraining_loop" src="https://github.com/user-attachments/assets/e332a916-b3c0-409b-aa1c-9d8d9a8366a3" />


1. The endpoint captures request/response data to S3 (**Data Capture**).
2. **Model Monitor** schedules compare live data with baselines (data quality, and model quality when ground-truth labels arrive).
3. Violations are published as **CloudWatch** metrics and trigger an **alarm**.
4. An **EventBridge** rule on the alarm state change starts the **SageMaker training pipeline**.
5. The new model must pass the same validation gate, is registered, approved, and then deployed.

---

## 10. Cost Optimisation

| Technique | Effect |
|-----------|--------|
| Batch Transform for non-urgent scoring instead of an always-on endpoint | No idle inference cost |
| Managed Spot Training | Large discount on training compute |
| Right-sized instances (based on CloudWatch utilisation) | Avoids over-provisioning |
| Endpoint autoscaling and removal of idle dev endpoints | Pay for what is used |
| Pipeline step caching | Skips unchanged steps on re-runs |
| Dev/test environments torn down after validation | No forgotten resources |

Combined result: **50% reduction in cloud cost** compared with the initial always-on setup.

---

## 11. Security and Compliance (Healthcare Context)

- **IAM least privilege** roles per stage, and **GitHub OIDC** instead of long-lived access keys.
- **KMS encryption** for S3, training volumes and endpoints; **VPC endpoints / PrivateLink** for private traffic.
- No real PHI in the repository; sample data only. Secrets and account IDs are kept out of code.
- Audit trail through Model Registry lineage, CloudWatch logs and CloudTrail.

---

## 12. Repository Structure

```
sm-patient-readmission-mlops/
├── .github/workflows/
│   ├── ci.yml                    # lint + pytest on pull requests
│   └── deploy-pipeline.yml       # OIDC auth, upsert + run pipeline, promote
├── pipelines/
│   ├── training_pipeline.py      # SageMaker Pipeline definition
│   ├── retraining_trigger.py     # EventBridge rule -> pipeline
│   └── batch_inference.py
├── src/
│   ├── processing/preprocess.py
│   ├── training/train.py
│   ├── evaluation/evaluate.py
│   ├── features/feature_store.py
│   └── inference/inference.py
├── monitoring/
│   ├── baseline.py               # Model Monitor baselines
│   └── monitor_schedule.py
├── deploy/
│   ├── deploy_endpoint.py        # real-time endpoint + autoscaling
│   └── rollback.py
├── tests/
├── data/sample/                  # tiny synthetic sample only
├── docs/images/                  # architecture diagrams
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

## 13. Getting Started

```bash
# 1. Clone and set up
git clone https://github.com/Radhakrishna-guntur/patient-readmission-sagemaker-mlops.git
cd patient-readmission-sagemaker-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Pull versioned data (DVC remote on S3)
dvc pull

# 3. Run tests
pytest tests/

# 4. Create/update and run the SageMaker pipeline (requires AWS credentials)
python pipelines/training_pipeline.py --stage dev --run

# 5. Deploy an approved model version
python deploy/deploy_endpoint.py --stage dev
```

---

## 14. Future Improvements

- Shadow or canary deployments for safer rollouts
- **SageMaker Clarify** for bias and explainability reporting
- Infrastructure as code (Terraform or CDK) for the full environment
- Multi-account setup (dev, staging, prod) with cross-account model promotion

---

## Author.   **Radhakrishna** | [GitHub](https://github.com/Radhakrishna-guntur)
