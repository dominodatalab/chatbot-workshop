# Chatbot Governance Workshop App

Welcome to the **Chatbot Governance Workshop** repository! This app is designed to support a hands-on governance workshop using Domino Data Lab, centered around model lifecycle management, policy enforcement, and auditability—all through a practical chatbot example.

## Workshop Context

This workshop guides you through modern model governance practices with Domino, referencing Fitch's model validation and documentation standards. You'll learn how models are developed, governed, and validated in Domino, using the HelpBot chatbot as the running example. The flow covers both the model developer and model validator perspectives, ensuring you see every step of the governance journey.

## Table of Contents

- [Workshop Outline](#workshop-outline)
- [App Structure](#app-structure)
- [Governance Concepts](#governance-concepts)
- [Running the HelpBot App](#running-the-helpbot-app)
- [Model Governance in Domino](#model-governance-in-domino)
- [Validation and Audit](#validation-and-audit)
- [References](#references)

---

## Workshop Outline

The workshop is structured as follows:

### 1. Domino Core Concepts

- Quick overview: **environments**, **projects**, **workspaces**, **experiments**, **models**, and **apps**.
- Introduction to Domino's governance capabilities.

### 2. Fitch Governance Practices

- Review of Fitch’s original governance docs (spreadsheets, documentation).
- Domino Policies: Overview and mapping of Fitch policies (General Intake, 3rd Party Intake, Final Documentation).
- Policy Editor UI demonstration.

### 3. Bundles and Model Governance Dashboard

- Understand Bundles in Domino.
- See how Fitch’s documentation maps to Domino Bundles.
- Explore the Model Manager/Governance Dashboard.

### 4. Model Change and Governance Review

- Kick off a model governance review with a new HelpBot bundle ("HelpBot Model Update").
- Copy evidence from previous bundles.
- Assign stages and approvals.

### 5. Developer Experience: Model Update

- Review app structure and environment variables.
- Use custom pip packages and configure the project environment.
- Workspace setup, code edit, and sync.
- Integration with Arize for model monitoring and telemetry.
- Experiments and version tracking via MLflow.
- Register and attach new model versions to bundles.
- Running HelpBot as an app.

### 6. Validator Experience: Model Validation

- Spin up workspace, run the HelpBot locally (`PORT=8501 bash app.sh`).
- Verify integration with Arize and Domino Experiments.
- Audit trail demonstration, report download, and runtime environment checks.
- Scripted checks and governance PDF generation.
- Address findings, upload audit reports, approve and publish the bundle.

---

## App Structure

- **`app.sh`**: Launch script for the HelpBot application.
- **Environment Variables**: Used for app configuration, API keys, and integrations (e.g., Arize).
- **Custom pip packages**: Required for HelpBot; see `requirements.txt` for details.
- **Arize Integration**: Telemetry and monitoring.
- **MLflow**: Model experiments and versioning.

## Governance Concepts

- **Policies**: Enforced via Domino’s policy engine. Examples include General Intake, 3rd Party Intake, and Final Documentation.
- **Bundles**: Logical groupings of model assets, documentation, and evidence.
- **Audit Trails**: Full traceability for model changes, validations, and approvals.

## Running the HelpBot App

To run HelpBot locally:

```bash
PORT=8501 bash app.sh
```

- Make sure required environment variables are set (see `.env.example`).
- Confirm all pip dependencies are installed (`pip install -r requirements.txt`).
- The app will be accessible at `http://localhost:8501`.

## Model Governance in Domino

- **Initiate Governance Review**: Create a new HelpBot bundle and copy evidence from previous bundles.
- **Assign Stages**: Route for relevant approval (e.g., to an org-wide group).
- **Register New Model Version**: Use MLflow experiment results for versioning.
- **Attach Model to Bundle**: Link the model artifact to the active governance bundle.

## Validation and Audit

- **Workspace Validation**: Run the model locally, validate telemetry and experiment logs.
- **Audit Reporting**: Download detailed audit reports for compliance.
- **Scripted Checks**: Automated governance documentation and runtime environment validation.
- **Findings Management**: Log, address, and resolve findings with supporting evidence (PDFs, logs).

## References

- [Domino Data Lab Documentation](https://docs.dominodatalab.com/)
- [Arize AI Model Monitoring](https://arize.com/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

## Contributing

If you have feedback, suggestions, or want to help improve the workshop materials, please open an issue or submit a pull request!

---
