import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json

print("=" * 60)
print("STEP 1: LOADING & CLEANING DATA")
print("=" * 60)

#load data
df = pd.read_csv('train.csv')

#Drop the unnamed index column
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print(f"✅ Loaded {len(df)} car listings")

#Extract numerical values from string columns
def extract_number(series):
    
    return pd.to_numeric(
        series.astype(str).str.extract(r'([\d.]+)')[0], 
        errors='coerce'
    )

df['Mileage'] = extract_number(df['Mileage'])
df['Engine'] = extract_number(df['Engine'])
df['Power'] = extract_number(df['Power'])

print("✅ Extracted numerical values from Mileage, Engine, Power")

# Extract Brand from Name
df['Brand'] = df['Name'].apply(lambda x: str(x).split(' ')[0])
print(f"✅ Extracted {df['Brand'].nunique()} unique car brands")

#Calculate Car Age
CURRENT_YEAR = 2026  # Update this if running in a different year
df['Car_Age'] = CURRENT_YEAR - df['Year']
print(f"✅ Created Car_Age feature (range: {df['Car_Age'].min()} to {df['Car_Age'].max()} years)")

#Handle Missing Values
print(f"\n📊 Missing values BEFORE cleaning:")
numeric_cols = ['Mileage', 'Engine', 'Power', 'Seats']
for col in numeric_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"   {col}: {missing_count} missing → filled with median ({median_val})")

# Drop rows where Price is missing
df = df.dropna(subset=['Price'])
print(f"\n✅ Cleaned dataset: {len(df)} rows remaining")

print("\n" + "=" * 60)
print("STEP 2: ENCODING CATEGORICAL FEATURES")
print("=" * 60)

label_encoders = {}
categorical_cols = ['Brand', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']

#Label Encoding
for col in categorical_cols:
    le = LabelEncoder()
    # Fill any NaN with 'Unknown' before encoding
    df[col] = df[col].fillna('Unknown')
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"✅ Encoded {col}: {len(le.classes_)} categories")
    # Show first few mappings as example
    mapping = dict(zip(le.classes_[:5], le.transform(le.classes_[:5])))
    print(f"   Example: {mapping}")

print("\n" + "=" * 60)
print("STEP 3: PREPARING FEATURES FOR THE MODEL")
print("=" * 60)

#Select Features
feature_columns = [
    'Car_Age',              # How old the car is
    'Kilometers_Driven',    # How much the car has been driven 
    'Mileage',              # Fuel efficiency
    'Engine',               # Engine size in CC
    'Power',                # Engine power in BHP
    'Seats',                # Number of seats
    'Brand_encoded',        # Car brand (encoded)
    'Location_encoded',     # City (encoded)
    'Fuel_Type_encoded',    # Fuel type (encoded)
    'Transmission_encoded', # Manual/Automatic (encoded)
    'Owner_Type_encoded',   # First/Second/Third owner (encoded)
]

X = df[feature_columns]
y = df['Price']  # Target variable — what we want to predict

print(f"✅ Feature matrix shape: {X.shape}")
print(f"   → {X.shape[0]} samples (cars), {X.shape[1]} features each")
print(f"\n📋 Features used:")
for i, col in enumerate(feature_columns, 1):
    print(f"   {i:2d}. {col}")

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,     # 20% for testing, 80% for training
    random_state=42     # For reproducibility
)

print(f"\n✅ Data split:")
print(f"   Training set: {len(X_train)} samples (80%)")
print(f"   Testing set:  {len(X_test)} samples (20%)")

print("\n" + "=" * 60)
print("STEP 4: TRAINING THE RANDOM FOREST MODEL")
print("=" * 60)

#Random Forest
print("🔄 Training Random Forest Regressor...")
print("   (This is a REGRESSION model because we predict a continuous value: price)")

model = RandomForestRegressor(
    n_estimators=200,     # 200 decision trees in our "forest"
    max_depth=15,         # Each tree can be at most 15 levels deep
    min_samples_split=5,  # Need at least 5 samples to split a node
    min_samples_leaf=2,   # Each leaf must have at least 2 samples
    random_state=42,      # Reproducibility
    n_jobs=-1             # Use ALL CPU cores for faster training
)

# .fit() is where the actual LEARNING happens
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

print("\n" + "=" * 60)
print("STEP 5: EVALUATING THE MODEL")
print("=" * 60)

#Make predictions on test data
y_pred = model.predict(X_test)

#Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"""
📊 Model Performance Metrics:
{'─' * 40}
  R² Score:  {r2:.4f}  (1.0 = perfect, 0.0 = useless)
  MAE:       ₹{mae:.2f} Lakhs  (average error magnitude)
  RMSE:      ₹{rmse:.2f} Lakhs  (penalizes large errors more)

💡 WHAT THESE MEAN:
  • R² of {r2:.2f} means the model explains {r2*100:.1f}% of price variation
  • MAE of {mae:.2f} means predictions are off by ₹{mae:.2f}L on average
  • RMSE is higher than MAE because it punishes big mistakes more heavily
""")

#Feature Importance
print("🏆 Feature Importance Ranking:")
print("   (Which features the model relies on most to predict price)")
print(f"{'─' * 50}")

importances = model.feature_importances_
feature_imp = pd.Series(importances, index=feature_columns)
feature_imp = feature_imp.sort_values(ascending=False)

for rank, (feature, importance) in enumerate(feature_imp.items(), 1):
    bar = '█' * int(importance * 50)
    print(f"   {rank:2d}. {feature:25s} {importance:.4f} {bar}")

print("\n" + "=" * 60)
print("STEP 6: GENERATING PRICE LABELS")
print("=" * 60)

#Predict fair value for ALL data
all_predictions = model.predict(X)
df['Predicted_Price'] = all_predictions

#Calculate price deviation
df['Price_Deviation'] = (df['Price'] - df['Predicted_Price']) / df['Predicted_Price']

#Assign labels
THRESHOLD = 0.15  # 15% tolerance

def get_label(deviation):
    """Classify a car based on how much its price deviates from fair value"""
    if deviation > THRESHOLD:
        return 'Overpriced'
    elif deviation < -THRESHOLD:
        return 'Underpriced'
    else:
        return 'Fair Price'

df['Label'] = df['Price_Deviation'].apply(get_label)

# Show distribution of labels
label_counts = df['Label'].value_counts()
print(f"\n📊 Label Distribution:")
for label, count in label_counts.items():
    pct = (count / len(df)) * 100
    emoji = {'Overpriced': '🔴', 'Underpriced': '🟢', 'Fair Price': '🟡'}
    print(f"   {emoji.get(label, '⚪')} {label:12s}: {count:5d} ({pct:.1f}%)")

print(f"\n💡 HOW TO INTERPRET:")
print(f"   A car is 'Overpriced' if its listed price is >{THRESHOLD*100:.0f}% above what the model predicts")
print(f"   A car is 'Underpriced' if its listed price is >{THRESHOLD*100:.0f}% below what the model predicts")
print(f"   'Fair Price' means it's within ±{THRESHOLD*100:.0f}% of the predicted value")

print("\n" + "=" * 60)
print("STEP 7: SAVING THE MODEL")  
print("=" * 60)

#Create model directory
os.makedirs('model', exist_ok=True)

#Save the trained model using joblib
joblib.dump(model, 'model/car_price_model.pkl')
print("✅ Saved: model/car_price_model.pkl")

#Save the label encoders
joblib.dump(label_encoders, 'model/label_encoders.pkl')
print("✅ Saved: model/label_encoders.pkl")

#Save feature columns list
model_config = {
    'feature_columns': feature_columns,
    'categorical_cols': categorical_cols,
    'threshold': THRESHOLD,
    'current_year': CURRENT_YEAR,
    'r2_score': round(r2, 4),
    'mae': round(mae, 2),
    'brands': sorted(label_encoders['Brand'].classes_.tolist()),
    'locations': sorted(label_encoders['Location'].classes_.tolist()),
    'fuel_types': sorted(label_encoders['Fuel_Type'].classes_.tolist()),
    'transmissions': sorted(label_encoders['Transmission'].classes_.tolist()),
    'owner_types': sorted(label_encoders['Owner_Type'].classes_.tolist()),
}

with open('model/model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)
print("✅ Saved: model/model_config.json")

#Save some example predictions for verification
print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS (first 5 test samples)")
print("=" * 60)

sample_df = df.iloc[:5][['Name', 'Year', 'Price', 'Predicted_Price', 'Label']].copy()
sample_df['Predicted_Price'] = sample_df['Predicted_Price'].round(2)
print(sample_df.to_string(index=False))

print(f"""
{'=' * 60}
🎉 TRAINING COMPLETE!
{'=' * 60}

📁 Files saved:
   model/car_price_model.pkl   — The trained ML model
   model/label_encoders.pkl    — Category-to-number mappings
   model/model_config.json     — Configuration & metadata

🔜 NEXT STEP: Run 'python app.py' to start the web application!
""")