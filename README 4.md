# Week 4: Predictive Modeling and Optimization in Logistics

Predicts shipment delivery time (hours) with four regression models and uses the selected model to test operational changes.

## Contents

| File / folder | Description |
|---|---|
| `predictive_model.py` | Full pipeline: simulation, preprocessing, model comparison, tuning, evaluation, optimisation scenarios |
| `report.docx` | Written report: problem definition, models, metrics, results, recommendations |
| `charts/` | Five figures used in the report |
| `results/` | CSV and JSON files with all numeric results |
| `data/logistics_shipments.csv` | Simulated dataset (5,000 shipments) |

## Setup and run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python predictive_model.py
```

The script takes about 2 to 3 minutes on a single CPU, because it runs grid searches with cross-validation. It regenerates the dataset, results, and charts, using a fixed random seed (42), so results are reproducible.

## Main results

- **Selected model:** Gradient Boosting (lowest 5-fold cross-validation RMSE, 4.35 h)
- **Test set:** RMSE 4.16 h, MAE 3.34 h, R² 0.944 (baseline predicting the average: RMSE 17.65 h)
- **Strongest levers from what-if analysis:** reduce weather exposure (-2.1 h), reduce stops per route (-1.9 h), keep warehouse load below 85% (-1.5 h overall)

The data is simulated from a known formula, so these accuracy figures show how well the method recovers that formula. They should be confirmed on real records.

## Author

Name: ____________________   Roll number: ____________________
