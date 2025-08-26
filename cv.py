import pandas as pd
import numpy as np
import gdown
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import gc

# ============================== Load data ==============================
url = 'https://drive.google.com/uc?id=193Zf0UVuzb2Gimje8UJlJ7x-TcEmWXYT'
output = 'crime_data.csv'
gdown.download(url, output, quiet=False)
df = pd.read_csv(output, encoding='utf-8', low_memory=False)

# ============================== Data Cleaning =========================
mean_value = df['Vict Age'].mean()
df['Vict Age'] = df['Vict Age'].apply(lambda x: mean_value if x <= 0 else x)
df['Vict Sex'].fillna('X', inplace=True)
df['Vict Descent'].fillna('X', inplace=True)
df["Premis Cd"] = df["Premis Cd"].fillna(0)
non_weapon_values = ['LIQUOR/DRUGS','VERBAL THREAT','PHYSICAL PRESENCE','DEMAND NOTE','BOMB THREAT',np.nan]
df['Weapon Used'] = df['Weapon Desc'].isin(non_weapon_values).astype(int)

def categorize_by_type(row):
    description = str(row['Crm Cd Desc']).lower()
    if any(word in description for word in ['burglary','fraud','vandalism','arson','embezzlement','forgery','counterfeit','identity','bunco','shoplifting','till tap','coin machine','document','property','looting']):
        return 'Property Crimes'
    elif any(word in description for word in ['assault','battery','homicide','manslaughter','lynching','intimate partner','resisting arrest','child abuse','fight']):
        if any(word in description for word in ['sexual','rape','lewd','sodomy','molest','penetration','oral copulation','lascivious','sex','indecent','unlawful']):
            return 'Sexual Assault'
        return 'Assault'
    elif any(word in description for word in ['theft','robbery','stolen','shoplifting','snatching','pickpocket','drunk roll','petty','grand','attempted theft','embezzlement']):
        return 'Theft'
    else:
        return 'Other'

df['Category'] = df.apply(categorize_by_type, axis=1)
area_to_bureau = {1:'Central',2:'Central',3:'South',4:'Central',5:'South',6:'West',7:'West',8:'West',9:'Valley',10:'Valley',11:'Central',12:'South',13:'Central',14:'West',15:'Valley',16:'Valley',17:'Valley',18:'South',19:'Valley',20:'West',21:'Valley'}
df['BUREAU'] = df['AREA'].map(area_to_bureau)

irrelevant_cols = ['DR_NO','Rpt Dist No','Mocodes','Premis Desc','Weapon Used Cd','Weapon Desc','Status','Status Desc','Part 1-2','Crm Cd 1','Crm Cd 2','Crm Cd 3','Crm Cd 4','Cross Street','Crm Cd Desc','AREA NAME','LOCATION','LAT','LON']
df_cleaned = df.drop(columns=irrelevant_cols, errors='ignore')

# ============================== Model & Feature Selection =========================
models = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42), {'C':[0.1,1,10]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {'max_depth':[5,10,20], 'min_samples_split':[2,5]}),
    "Random Forest": (RandomForestClassifier(random_state=42), {'n_estimators':[100,200], 'max_depth':[10,20], 'min_samples_split':[2,5]}),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), {'n_estimators':[100,200], 'max_depth':[3,6], 'learning_rate':[0.01,0.1]}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors':[3,5,7]}),
    "SVM": (LinearSVC(max_iter=5000, random_state=42), {'C':[0.1,1,10]})
}

feature_counts = [5, 6, 7]
targets = {
    "Category": [col for col in df_cleaned.columns if col not in ['BUREAU','Category','Crm Cd']],
    "BUREAU": [col for col in df_cleaned.columns if col not in ['BUREAU','Category','AREA']]
}

# ============================== Results storage =========================
cv_results_list = []
validation_results_list = []

for target_col, candidate_features in targets.items():
    print(f"\n{'='*50}\nTarget: {target_col}\n{'='*50}")
    df_target = df_cleaned[[target_col]+candidate_features].copy()

    df_encoded = df_target.copy()
    label_encoders = {}
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    features = [col for col in df_encoded.columns if col != target_col]
    X = df_encoded[features]
    y = df_encoded[target_col]

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_val_score = 0
    best_model_info = None

    for k in feature_counts + [X_train.shape[1]]:
        k = min(k, X_train.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)
        selected_cols = X_train.columns[selector.get_support()]
        X_train_selected = X_train[selected_cols]
        X_val_selected = X_val[selected_cols]

        balancing_methods = {
            'RandomOverSampler': RandomOverSampler(random_state=42),
            'SMOTE': SMOTE(random_state=42),
            'UnderSampling': RandomUnderSampler(sampling_strategy='not minority', random_state=42)
        }

        for method_name, sampler in balancing_methods.items():
            X_bal, y_bal = sampler.fit_resample(X_train_selected, y_train)

            for model_name, (model, param_grid) in models.items():
                grid = GridSearchCV(model, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
                grid.fit(X_bal, y_bal)
                best_estimator = grid.best_estimator_
                val_score = grid.best_score_

                cv_results_list.append({
                    "Target": target_col,
                    "Num Features": k,
                    "Balancing Method": method_name,
                    "Model": model_name,
                    "Validation Score": val_score,
                    "Selected Features": ', '.join(selected_cols)
                })

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model_info = {
                        'model_name': model_name,
                        'model': best_estimator,
                        'selected_cols': selected_cols,
                        'sampler': sampler,
                        'feature_count': k,
                        'balancing_method': method_name
                    }

    # Train best model on full train+val
    sampler = best_model_info['sampler']
    selected_cols = best_model_info['selected_cols']
    model = best_model_info['model']
    X_train_val_selected = pd.concat([X_train[selected_cols], X_val[selected_cols]])
    y_train_val_full = pd.concat([y_train, y_val])
    X_train_val_bal, y_train_val_bal = sampler.fit_resample(X_train_val_selected, y_train_val_full)
    model.fit(X_train_val_bal, y_train_val_bal)

    # Test evaluation
    X_test_selected = X_test[selected_cols]
    y_pred_test = model.predict(X_test_selected)

    metrics_dict = {
        "Target": target_col,
        "Model": best_model_info['model_name'],
        "Balancing Method": best_model_info['balancing_method'],
        "Num Features": best_model_info['feature_count'],
        "Accuracy": accuracy_score(y_test, y_pred_test),
        "F1 Macro": f1_score(y_test, y_pred_test, average='macro'),
        "Precision Macro": precision_score(y_test, y_pred_test, average='macro'),
        "Recall Macro": recall_score(y_test, y_pred_test, average='macro'),
        "Selected Features": ', '.join(selected_cols)
    }
    validation_results_list.append(metrics_dict)

    print(f"\n=== Test Results for {target_col} ===")
    print(classification_report(y_test, y_pred_test))

    # Save model & encoders
    joblib.dump({"model": model}, f"best_model_{target_col}.pkl")
    joblib.dump(label_encoders, f"encoders_{target_col}.pkl", compress=('xz',3))
    print(f"✅ Model and encoders saved for {target_col}")

    # Cleanup
    del X, y, X_train, X_val, X_test, y_train, y_val, model, y_pred_test
    gc.collect()

# ============================== Save CV & Validation Results =========================
cv_results_df = pd.DataFrame(cv_results_list)
cv_results_df.to_csv("cv_results.csv", index=False)
print("✅ CV results saved to cv_results.csv")

val_results_df = pd.DataFrame(validation_results_list)
val_results_df.to_csv("validation_results.csv", index=False)
print("✅ Validation results saved to validation_results.csv")
