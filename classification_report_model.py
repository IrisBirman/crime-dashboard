import pandas as pd
import numpy as np
import gdown
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# === שלב 1: טעינת הנתונים ===# ============================== Load data ==============================
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


# === שלב 3: טעינת encoders ==targets_info = {
targets_info = {    
    "Category": {
        "features": ["Date Rptd","DATE OCC","TIME OCC","AREA","Vict Age","Vict Sex","Vict Descent","Premis Cd","Weapon Used"],
        "model_file": "best_model_Category.pkl",
        "encoders_file": "encoders_Category1.pkl"
    },
    "BUREAU": {
        "features": ["Date Rptd","DATE OCC","TIME OCC","Crm Cd","Vict Age","Vict Sex","Vict Descent","Premis Cd","Weapon Used"],
        "model_file": "best_model_BUREAU.pkl",
        "encoders_file": "encoders_BUREAU1.pkl"
    }
}

# === שלב 4: חלוקה ובדיקת מודלים ===
for target_col, info in targets_info.items():
    print(f"\n===== Evaluating model for target: {target_col} =====")

    # טעינת הקידודים השמורים
    label_encoders = joblib.load(info["encoders_file"])

    # יצירת עותק מקודד של הנתונים
    df_encoded = df_cleaned.copy()
    for col, le in label_encoders.items():
        if col in df_encoded.columns:
            df_encoded[col] = le.transform(df_encoded[col].astype(str))

    # יצירת X ו-y
    X = df_encoded[info["features"]]
    y = df_encoded[target_col]

    # חלוקה זהה ל-seed המקורי
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1111, random_state=42, stratify=y_train_val
    )

    # טעינת המודל השמור
    model = joblib.load(info["model_file"])["model"]

    # חיזוי על קבוצת ה-test
    y_pred = model.predict(X_test)

    # יצירת דוח
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # הדפסה למסך
    print(report_df)

    # שמירה לקובץ CSV
    csv_name = f"{target_col}_test_results.csv"
    report_df.to_csv(csv_name, index=True)
    print(f"✅ Test results saved to: {csv_name}")

