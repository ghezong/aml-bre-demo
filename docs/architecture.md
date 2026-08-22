# Architecture

## Components

```mermaid
flowchart TD
    A[Transaction CSV] --> B[Schema validation and date parsing]
    B --> C[Rule feature builder]
    C --> D1[Many-to-One count]
    C --> D2[One-to-Many count]
    C --> D3[Mule pair total]
    C --> D4[Dormancy gap]
    D1 --> E[Feature matrix]
    D2 --> E
    D3 --> E
    D4 --> E
    E --> F{Training or scoring?}
    F -->|Training + alert label| G[Stratified validation split]
    G --> H[StandardScaler + class-balanced LogisticRegression]
    H --> I[Learn F1 operating threshold]
    I --> J[Serialized model bundle]
    F -->|New data| K[Probability scoring]
    J --> K
    K --> L[Prediction + rule contributions]
    L --> M[Streamlit review and CSV export]
```

## Boundaries

- `src/aml_core/aml_rules.py` owns deterministic rule features and screening rules.
- `src/aml_core/ml_model.py` owns preprocessing, training, threshold selection, scoring, persistence, and explainability.
- `src/ui/streamlit_app.py` owns user interaction and visual review.
- `data/` contains synthetic examples only; `models/` and `outputs/` are runtime locations.

The model bundle contains the fitted pipeline, feature names, learned coefficients, reference levels, and evaluation summary. In a production system this would also carry dataset identifiers, code version, training timestamp, approval status, and lineage metadata.
