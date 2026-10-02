# IMLP4 Task 03 – Product Category Prediction

## Objective
Build a machine-learning system that predicts a product category from its title. The task specification requires data analysis and cleaning, feature engineering, comparison of at least two ML models, a saved `.pkl` model, `train_model.py`, `predict_category.py`, a notebook, and clear documentation.

## Dataset
`IMLP4_TASK_03-products.csv` contains 35,311 rows and 8 columns. The target is `Category Label`; the main input is `Product Title`.

The raw data contains small label variants such as `fridge`, `CPU`, and `Mobile Phone`. They are normalized to `Fridge Freezers`, `CPUs`, and `Mobile Phones`.

## Approach
1. Load and inspect the CSV.
2. Strip whitespace from column names.
3. Remove rows with missing title/target and empty titles.
4. Normalize category labels.
5. Split data into 80% train / 20% test using stratification.
6. Compare:
   - Multinomial Naive Bayes with word TF-IDF.
   - Logistic Regression with word TF-IDF.
   - Linear SVM (LinearSVC) with combined word and character TF-IDF.
7. Evaluate accuracy, precision, recall, F1-score and confusion matrix.
8. Train the final LinearSVC model on all cleaned data and save `model.pkl`.
9. Use `predict_category.py` for interactive prediction.

## Feature engineering
The main textual representation uses:
- word TF-IDF, n-grams 1–2;
- character TF-IDF with `char_wb`, n-grams 3–5.

Character n-grams help capture product/model fragments, abbreviations, spelling variations and manufacturer/model patterns.

## Validation result
On the fixed 20% stratified test split (random_state=42):
- MultinomialNB: ~95.95% accuracy
- LogisticRegression: ~95.75% accuracy
- LinearSVC with word + character TF-IDF: ~98.76% accuracy

The final `model.pkl` is trained on the complete cleaned dataset after evaluation.

## Project structure
```text
.
├── IMLP4_TASK_03-products.csv
├── model.pkl
├── train_model.py
├── predict_category.py
├── product_category_classification.ipynb
├── requirements.txt
└── README.md
```

## Google Colab
Upload the CSV and notebook to Colab, then run all cells from top to bottom.

Or install dependencies:
```bash
pip install -r requirements.txt
```

Train:
```bash
python train_model.py
```

Interactive prediction:
```bash
python predict_category.py
```

Single title:
```bash
python predict_category.py --title "apple iphone 7 32gb gold"
```

## Example test titles from the assignment
Expected categories listed by the task:
- `iphone 7 32gb gold` → Mobile Phones
- `olympus e m10 mark iii geh use silber` → Digital Cameras
- `kenwood k20mss15 solo` → Microwaves
- `bosch wap28390gb 8kg 1400 spin` → Washing Machines
- `bosch serie 4 kgv39vl31g` → Fridge Freezers
- `smeg sbs8004po` → Fridge Freezers

## Reproducibility
The evaluation uses `random_state=42` and stratified splitting. The final model is retrained on all cleaned rows.
