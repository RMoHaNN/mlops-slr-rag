# RQ2 — What design principles for Industrial AI can be identified from the gaps between the two literatures?

_Each principle addresses a verified cross-stream constraint difference (RQ1 categories a/b) and is supported by verbatim-verified quotes from the per-paper gap extractions. Principles with fewer than 2 supporting papers are labelled **single-source**._

## P1. Implement robust data quality and governance mechanisms tailored for industrial data characteristics.

**Constraint difference addressed:** Data Quality and Availability

**Rationale:** The industrial stream emphasizes challenges such as data being scattered, inconsistent, imbalanced, sparse, noisy, and having missing values, along with manual collection and privacy restrictions. Many existing data quality tools are no longer completely supported, and there is a lack of empirical studies investigating how practitioners navigate data quality issues in applied settings. Without high-quality, labeled data, model performance suffers, making deployment in manufacturing environments particularly challenging.

**What to do:** Design and integrate automated data quality checks, validation thresholds, and data governance processes into MLOps pipelines. Prioritize data collection strategies that ensure high-quality, labeled data, and develop specialized tools for preprocessing technical texts and handling domain-specific data issues like sensor noise and missing values. Ensure data privacy and confidentiality are maintained through robust schemes and access controls.

**Supporting evidence:**
- `10.1002/smr.70044`: "However, a significant number of practitioners do not currently use data quality checks or measurements as gateways for their model construction and operationalization, indicating a need for greater awareness and adoption of these tools."
- `10.1002/smr.70044`: "While significant advancements have been made in addressing data quality challenges in ML, there is a lack of empirical studies investigating how practitioners navigate data quality issues in applied settings, such as those encountered in ML pipelines."
- `10.1080/21693277.2026.2658878`: "Another significant hurdle is related to data availability and quality. Data collection and preprocessing consume a significant portion of resources in ML projects. Without high-quality, labeled data, model performance suffers, making deployment in manufacturing environments particularly challenging (Mayr et al., 2019 )."
- `10.1007/978-3-031-25448-2_5`: "V ery little attention is given to how difﬁcult it is to curate meaningful data sets, especially for in-service assets. Most illustrations come from accelerated life tests or other well-resourced reliability programs on critical assets."
- `10.1016/j.heliyon.2025.e44416`: "While existing approaches treat data quality assessment and ML systems as isolated processes, our framework ad-dresses the critical gap between theoretical methods and practical implemen-tation by combining dynamic drift detection, adaptive data quality metrics, and MLOps into a cohesive, lightweight system."

## P2. Design for seamless integration into existing industrial infrastructure and legacy systems.

**Constraint difference addressed:** Model Deployment and Integration Complexity

**Rationale:** The industrial stream highlights the complexity of integrating into existing industrial infrastructure and legacy systems, requiring careful architectural design and compatibility with diverse hardware and software. Traditional software development methodologies often struggle with the iterative, nonlinear nature of ML workflows, leading to inefficiencies and slow delivery cycles. Integrating ML models with legacy applications presents a challenge, and a typical MLOps pipeline does not directly cater to legacy systems.

**What to do:** Develop MLOps solutions with modular, scalable architectures that explicitly account for integration with brownfield systems and legacy hardware/software. Prioritize open and flexible software platforms for proper IT integration and process improvement. Ensure compatibility with diverse operating systems and processor architectures, and design for minimal interference with existing processes on shared legacy platforms.

**Supporting evidence:**
- `10.1007/978-3-031-56281-5_7`: "Reported problems include difficulties in designing the architecture of the infrastructure for production deployment and legacy application integration."
- `10.1007/978-3-032-07313-6_2`: "Faced with this abundance of options and trade-oﬀs, practitioners feel they lack guidance on MLOps in general and the integration step in particular [9– 11]."
- `10.1109/ETFA54631.2023.10275468`: "Also, a typical MLOps pipeline does not cater directly to legacy systems, and a customized inference pipeline is necessary for DNN deployment in such environments."
- `10.1109/ETFA54631.2023.10275468`: "The challenge, however, is that semiconductor manufacturing industries still use brownfield systems and equipment with legacy hardware and software. The legacy systems introduce challenging requirements and constraints on the DNN deployment and the traditional approach to inference optimization results in poor inference performance."
- `10.1016/j.jss.2025.112542`: "However, integrating EDAs with existing steel production infrastructure, which often relies on legacy systems, presents considerable technical challenges."

## P3. Optimize resource utilization and ensure scalability for constrained and distributed industrial environments.

**Constraint difference addressed:** Resource Constraints and Scalability

**Rationale:** The industrial stream specifically mentions limitations on edge devices and the impact on training complex models and handling large data volumes. Edge devices are limited in bandwidth, storage, and processing power, which can limit training complex ML models. Deploying deep learning models in real-world scenarios presents challenges beyond just training and evaluation, requiring continuous monitoring of their performance.

**What to do:** Develop lightweight and distributed ML models suitable for edge devices and resource-constrained environments. Implement dynamic resource orchestration and autoscaling policies that account for the volatile, stage-specific requirements of ML pipelines. Design for efficient data processing at the edge to reduce the burden on cloud resources and ensure system stability under high data throughput.

**Supporting evidence:**
- `10.1109/JIOT.2023.3268771`: "Edge devices are also limited in bandwidth, storage, and processing power, which can limit training complex ML models."
- `10.1109/ADACIS65663.2025.11437289`: "Nevertheless, native Kubernetes resource schedulers such as the Horizontal Pod Autoscaler (HPA) and Vertical Pod Autoscaler (VPA) are often designed with stateless microservices in mind and do not account for the volatile, stage-specific requirements of ML pipelines [2]."
- `10.1109/CSCN67557.2025.11230711`: "However, deploying all of these artifacts on edge devices raises concerns about the computational capacity of edge devices."
- _(2 further support item(s) dropped: quote could not be verified verbatim)_

## P4. Prioritize low-latency, real-time decision-making with robust and adaptive model updates.

**Constraint difference addressed:** Real-time and Latency Requirements

**Rationale:** The industrial stream emphasizes 'low-latency, real-time decision-making' in industrial applications, which is challenged by 'computationally intensive processes, data transmission delays, and the need for continuous data processing and model updates.' Static models' performance drops over time due to concept drift, requiring new models to be trained with recent data and redeployed in production. Drift detection must be performed in real-time to be effective, otherwise, the model's reaction may be too late.

**What to do:** Implement continuous learning approaches with dynamic model updates in real-time, rather than relying on batch ML with offline retraining. Integrate forecast-aware, drift-triggered retraining mechanisms into MLOps pipelines. Design systems to handle high-velocity sensor data prone to quality variations and ensure minimal latency in data ingestion and processing for timely decision-making.

**Supporting evidence:**
- `10.1016/j.compind.2022.103825`: "However, the most recently described method of testing dope polymer viscosity is at least 4 h too late for real-time decision-making, which would lead to belated efforts if any contingencies are present."
- `10.1109/ICDE55515.2023.00272`: "In industrial applications, static models’ performance drops over time (model degradation, concept drift), requiring new models to be trained with recent data and redeployed in production."
- `10.1109/ICDE55515.2023.00272`: "Continuously learning and serving from evolving streaming data and serving in real-time is a challenging problem."
- `10.1109/ICDMW53433.2021.00049`: "Moreover, drift detection should be performed in real-time to be effective. Otherwise, the model may react too late."
- `10.1109/BigDataService65758.2025.00019`: "Concept drift poses a critical challenge to deploying reliable machine learning models in real-world production environments, particularly in time series forecasting and predictive maintenance systems."

## P5. Embed security, privacy, and trust by design, especially in safety-critical industrial environments.

**Constraint difference addressed:** Security, Privacy, and Trust

**Rationale:** The industrial stream emphasizes 'safety-critical physical environments' and 'sensitive data, regulatory compliance, and the need for explainable and unbiased decisions,' including protection against adversarial attacks and accountability. Data records in their original form are often prohibited from being shared due to legal regulations like GDPR and HIPAA, as they contain sensitive information. Edge deployment aboard vessels amplifies the need for a secure pipeline, as compromised models or manipulated sensor data can undermine vessel safety.

**What to do:** Integrate robust cybersecurity measures, differential privacy, and federated learning approaches to protect sensitive data and intellectual property. Design MLOps systems to provide explainability by design, ensuring decisions made by AI models are traceable, interpretable, and justifiable. Implement automated policy enforcement and ensure compliance with relevant regulatory frameworks and industry standards.

**Supporting evidence:**
- `10.1007/978-3-031-96590-6_3`: "However, data records in their original form are often prohibited from being shared due to legal regulations such as GDPR and HIPAA, and as they often contain sensitive or exploitable information."
- `10.1109/ICIC68054.2025.11309501`: "Edge deployment also amplifies the need for a secure pipeline, as compromised models or manipulated sensor data can undermine vessel safety."
- `10.23919/CNSM59352.2023.10327814`: "This is especially true for industrial use cases, where the trust and reliability of ML applications are mission-critical."
- _(2 further support item(s) dropped: quote could not be verified verbatim)_

## P6. Develop tailored MLOps tooling and platforms that avoid vendor lock-in and support on-premises deployments.

**Constraint difference addressed:** Tooling and Platform Selection

**Rationale:** This constraint is unique to the industrial stream, focusing on the struggle with selecting appropriate MLOps tools and design options due to a proliferation of tools, avoiding vendor lock-in, and the lack of mature on-premises offerings, especially when cloud use is restricted. Organizations face a problem in selecting appropriate tools and design options for implementing MLOps environments due to the proliferation of tools and gray literature. Preventing vendor lock-in can be problematic as most organizations prioritize time-to-market over flexibility and portability when selecting a cloud provider or data/ML platform.

**What to do:** Prioritize the development and adoption of open-source, flexible MLOps frameworks that can be deployed on-premises and are not tied to specific cloud providers. Invest in tools that support reproducible packaging of ML models for isolated shop floor environments and provide clear guidelines for architectural decisions. Avoid solutions that lead to vendor lock-in and ensure that tooling supports the specific needs of industrial applications, including those with restricted cloud access.

**Supporting evidence:**
- `10.1007/978-3-032-02138-0_2`: "However, preventing vendor lock-in can be problematic as most organizations select and adopt a speciﬁc cloud provider or data/ML platform and prioritize time-to-market over ﬂexibility and portability (participant P5)."
- `10.1109/ETFA61755.2024.10711136`: "Further, it is important to note that commercial data platforms may lead to vendor lock-in, limiting flexibility in switching to alternative platforms if needed."
- _(3 further support item(s) dropped: quote could not be verified verbatim)_


## References (DOI — title, year, stream)

- `10.1002/smr.70044` — Engineering MLOps Pipelines With Data Quality: A Case Study on Tabular Datasets in Kaggle (2025, generic)
- `10.1007/978-3-031-25448-2_5` — RelOps – A Whole-of-Organisation Approach for Reliability Analytics (2023, generic)
- `10.1007/978-3-031-56281-5_7` — ML-Enabled Systems Model Deployment and Monitoring: Status Quo and Problems (2024, industrial)
- `10.1007/978-3-031-96590-6_3` — Data Chameleon: A Self-adaptive Synthetic Data Management System (2025, industrial)
- `10.1007/978-3-032-02138-0_2` — MLOps in Practice: Requirements and a Reference Architecture from Industry (2026, industrial)
- `10.1007/978-3-032-07313-6_2` — MLOps Adoption in the Manufacturing Industry: A Case Study with Zeiss SMT (2026, industrial)
- `10.1016/j.compind.2022.103825` — Development of a virtual metrology system for smart manufacturing: A case study of spandex fiber production (2023, industrial)
- `10.1016/j.heliyon.2025.e44416` — End-to-end data quality-driven framework for machine learning in production environment (2026, industrial)
- `10.1016/j.jss.2025.112542` — Smart manufacturing: MLOps-enabled event-driven architecture for enhanced control in steel production (2025, industrial)
- `10.1080/21693277.2026.2658878` — Industrial MLOps: a systematic review of architectures and implementation challenges (2026, industrial)
- `10.1109/ADACIS65663.2025.11437289` — MLPilot: An Intelligent CI/CD Flight Controller for Auto-Tuning Resource Allocation in MLOps Kubernetes Pipelines (2025, generic)
- `10.1109/BigDataService65758.2025.00019` — AutoDrift: A Forecast-Aware Concept Drift Detection and Retraining Pipeline in MLOps with CMAPSS (2025, industrial)
- `10.1109/CSCN67557.2025.11230711` — Leveraging AI and MLOps for IoT-Edge-Cloud Industrial Digital Twins: A Practical Case Study (2025, industrial)
- `10.1109/ETFA54631.2023.10275468` — A Structured Inference Optimization Approach for Vision-Based DNN Deployment on Legacy Systems (2023, industrial)
- `10.1109/ETFA61755.2024.10711136` — MLOps: A Multiple Case Study in Industry 4.0 (2024, industrial)
- `10.1109/ICDE55515.2023.00272` — StreamMLOps: Operationalizing Online Learning for Big Data Streaming & Real-Time Applications (2023, industrial)
- `10.1109/ICDMW53433.2021.00049` — Drift Lens: Real-time unsupervised Concept Drift detection by evaluating per-label embedding distributions (2021, generic)
- `10.1109/ICIC68054.2025.11309501` — Unified DevSecOps Toolchain for Secure AI-Enabled Maritime Applications: Theory, Concepts, and Implementation (2025, generic)
- `10.1109/JIOT.2023.3268771` — Seamless Transition From Machine Learning on the Cloud to Industrial Edge Devices With Thinger.io (2023, industrial)
- `10.23919/CNSM59352.2023.10327814` — Tailoring MLOps Techniques for Industry 5.0 Needs (2023, industrial)