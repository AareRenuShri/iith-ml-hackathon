
import pandas as pd
import unicodedata

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Training data:", train.shape)
print("Testing data:", test.shape)


# --------------------------------------------------
# 2. Feature extraction
# --------------------------------------------------

def extract_features(word):
    word = str(word)

    return {
        "lowercase": word.lower(),
        "word_length": len(word),
        "is_uppercase": word.isupper(),
        "is_digit": word.isdigit(),
        "prefix": word[:3],
        "suffix": word[-3:]
    }


# --------------------------------------------------
# 3. Prepare target labels
# --------------------------------------------------

target_parts = train["target"].str.split("|", n=1, expand=True)

train["head"] = target_parts[0].astype(int)
train["relation"] = target_parts[1]

label_encoder = LabelEncoder()
train["relation_label"] = label_encoder.fit_transform(
    train["relation"]
)

print("\nNumber of relation classes:", len(label_encoder.classes_))


# --------------------------------------------------
# 4. Prepare training and validation data
# --------------------------------------------------

X = [extract_features(word) for word in train["token"]]
y = train["relation_label"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = DictVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Numerical features:", X_train_vec.shape[1])


# --------------------------------------------------
# 5. Train Logistic Regression model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs"
)

print("\nTraining Logistic Regression model...")

model.fit(X_train_vec, y_train)

print("Training completed!")


# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------

y_pred = model.predict(X_val_vec)

validation_accuracy = accuracy_score(y_val, y_pred)
training_accuracy = accuracy_score(
    y_train,
    model.predict(X_train_vec)
)

majority_baseline = y_train.value_counts(normalize=True).max()

print("\n--- MODEL EVALUATION ---")
print(f"Training accuracy:   {training_accuracy:.4f}")
print(f"Validation accuracy: {validation_accuracy:.4f}")
print(f"Majority baseline:   {majority_baseline:.4f}")
print("Iterations used:", model.n_iter_[0])


# --------------------------------------------------
# 7. Generate test predictions
# --------------------------------------------------

X_test = [extract_features(word) for word in test["token"]]
X_test_vec = vectorizer.transform(X_test)

test_predictions = model.predict(X_test_vec)


# --------------------------------------------------
# 8. Generate relation-only submission
# --------------------------------------------------

submission_rel = pd.DataFrame({
    "ID": test["ID"],
    "Label": test_predictions
})

submission_rel.to_csv("submission_rel_baseline.csv", index=False)

print("\nRelation-only submission saved.")


# --------------------------------------------------
# 9. Predict head IDs
# --------------------------------------------------

def predict_head(token):
    token = str(token)

    if token and all(
        unicodedata.category(char).startswith("P")
        for char in token
    ):
        return 0

    return 1


head_predictions = test["token"].apply(predict_head)


# --------------------------------------------------
# 10. Generate head + relation submission
# --------------------------------------------------

combined_labels = [
    f"{head}|{relation}"
    for head, relation in zip(head_predictions, test_predictions)
]

submission_head_rel = pd.DataFrame({
    "ID": test["ID"],
    "Label": combined_labels
})

submission_head_rel.to_csv("submission_head_rel.csv", index=False)

print("Head + relation submission saved.")


# --------------------------------------------------
# 11. Final submission verification
# --------------------------------------------------

for filename, submission in [
    ("submission_rel_baseline.csv", submission_rel),
    ("submission_head_rel.csv", submission_head_rel)
]:
    print(f"\n{filename}")
    print("Rows:", len(submission))
    print("Columns:", list(submission.columns))
    print("Missing values:", submission.isna().sum().sum())

print("\nHead ID distribution:")
print(head_predictions.value_counts())

print("\nFirst 5 combined predictions:")
print(submission_head_rel.head())

print("\nPhase 2 model execution completed!")