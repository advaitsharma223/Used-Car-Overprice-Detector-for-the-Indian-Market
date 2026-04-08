# Used Car Price Evaluator — ML Project

Build an ML model that tells users whether a used car is **overpriced**, **underpriced**, or **fairly priced**, using your real [train.csv](file:///c:/Users/Advait/Desktop/ML%20Project1/train.csv) dataset.

> [!IMPORTANT]
> **Learning-first approach**: Every script will be **heavily commented** with explanations of *why* each step is done, not just what it does. You'll be able to explain every part of the pipeline.

## Your Dataset

| Column | Example | Notes |
|---|---|---|
| Name | `Hyundai Creta 1.6 CRDi SX` | Full car name |
| Location | `Pune` | City |
| Year | `2015` | Manufacturing year |
| Kilometers_Driven | `41000` | Odometer |
| Fuel_Type | `Diesel` / `Petrol` | Categorical |
| Transmission | `Manual` / `Automatic` | Categorical |
| Owner_Type | `First` / `Second` / `Third` | Categorical |
| Mileage | `19.67 kmpl` | Needs unit stripping |
| Engine | `1582 CC` | Needs unit stripping |
| Power | `126.2 bhp` | Needs unit stripping |
| Seats | `5.0` | Float → Int |
| New_Price | `8.61 Lakh` | Original price (many missing) |
| **Price** | `12.5` | **Target** — resale price in Lakhs |

---

## Approach

### Labeling Strategy (since we don't have pre-existing labels)
1. **Train a regression model** to predict a car's "fair value" from its features.
2. **Compare actual listed price vs. predicted fair value**:
   - **Overpriced** → listed ≥15% above predicted fair value
   - **Underpriced** → listed ≥15% below predicted fair value
   - **Fairly Priced** → within ±15%

### Why This Works
The model learns what a car *should* cost based on patterns in the data. Cars that deviate significantly from the predicted price are flagged. This is similar to how real pricing tools (KBB, CarDekho) work.

---

## Proposed Changes

### 1. Setup

#### [NEW] [requirements.txt](file:///c:/Users/Advait/Desktop/ML%20Project1/requirements.txt)
`pandas`, `numpy`, `scikit-learn`, `flask`, `flask-cors`, `joblib`, `matplotlib`, `seaborn`

---

### 2. Exploratory Data Analysis (EDA)

#### [NEW] [eda.py](file:///c:/Users/Advait/Desktop/ML%20Project1/eda.py)
- Loads [train.csv](file:///c:/Users/Advait/Desktop/ML%20Project1/train.csv), prints shape, dtypes, missing values
- Visualizes price distribution, price vs year, price vs km, brand-wise pricing
- Saves plots to `plots/` folder
- **Every step explained with comments** so you understand the data

---

### 3. Data Preprocessing + Model Training

#### [NEW] [train_model.py](file:///c:/Users/Advait/Desktop/ML%20Project1/train_model.py)
Step-by-step with comments explaining:
1. **Data Cleaning** — strip units from Mileage/Engine/Power, handle missing values
2. **Feature Engineering** — extract brand from Name, calculate car age
3. **Encoding** — Label encode categorical features (Brand, Fuel, Transmission, Owner)
4. **Train/Test Split** — 80/20 split
5. **Model Training** — Random Forest Regressor (predicts fair value)
6. **Evaluation** — R², MAE, RMSE printed and explained
7. **Label Generation** — compare actual vs predicted → assign Overpriced/Underpriced/Fair
8. **Save** — model + encoders to `model/` folder

---

### 4. Flask API

#### [NEW] [app.py](file:///c:/Users/Advait/Desktop/ML%20Project1/app.py)
- `/predict` POST → accepts car details, returns verdict + confidence
- `/health` GET → health check
- Commented to explain Flask routing, JSON handling, model loading

---

### 5. Web Frontend

#### [NEW] [templates/index.html](file:///c:/Users/Advait/Desktop/ML%20Project1/templates/index.html)
Premium dark-themed UI with glassmorphism, gradients, micro-animations

#### [NEW] [static/style.css](file:///c:/Users/Advait/Desktop/ML%20Project1/static/style.css)
Full design system with CSS variables and responsive layout

#### [NEW] [static/script.js](file:///c:/Users/Advait/Desktop/ML%20Project1/static/script.js)
Form submission → API call → animated result card

---

## Verification Plan

### Automated
1. Run `python eda.py` → verify plots saved to `plots/`
2. Run `python train_model.py` → verify R² ≥ 0.70, model saved to `model/`
3. Run `python app.py` → curl `/predict` → verify JSON response

### Manual
4. Open `http://localhost:5000` → fill form → verify prediction result
