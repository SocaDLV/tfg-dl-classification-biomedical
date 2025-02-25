import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (

    accuracy_score,

    confusion_matrix,

    classification_report,

    roc_curve,

    auc,

    precision_score,

    recall_score,

    f1_score,

)

from sklearn.model_selection import LeaveOneOut

import matplotlib.pyplot as plt

import seaborn as sns

from prettytable import PrettyTable

 

# Prepare the data

X_cancer = df_cancer_intersecting.drop(["Id", "Group"], axis=1)

X_control = df_control_intersecting.drop(["Id", "Group"], axis=1)

X = pd.concat([X_cancer, X_control])

y = ["cancer"] * X_cancer.shape[0] + ["control"] * X_control.shape[0]

 

# Initialize LOOCV and the Random Forest Classifier

loo = LeaveOneOut()

clf_loo = RandomForestClassifier(n_estimators=27, max_depth=3, random_state=42, class_weight="balanced")

 

# Store predictions, probabilities, and true labels

predictions = []

true_labels = []

y_probs = []

 

# Perform LOOCV

for train_index, test_index in loo.split(X):

    X_train_loo, X_test_loo = X.iloc[train_index], X.iloc[test_index]

    y_train_loo, y_test_loo = np.array(y)[train_index], np.array(y)[test_index]

 

    # Train the model and make predictions

    clf_loo.fit(X_train_loo, y_train_loo)

    y_pred_loo = clf_loo.predict(X_test_loo)[0]

    y_pred_prob_loo = clf_loo.predict_proba(X_test_loo)[0, 1]  # Probability of the positive class

 

    predictions.append(y_pred_loo)

    y_probs.append(y_pred_prob_loo)

    true_labels.append(1 if y_test_loo[0] == "control" else 0)  # Assuming "control" is the positive class

 

# Calculate the confusion matrix

conf_matrix_loo = confusion_matrix(y, predictions, labels=["control", "cancer"])

 

# Plot the confusion matrix using seaborn

plt.figure(figsize=(4, 3))

sns.heatmap(conf_matrix_loo, annot=True, fmt="d", cmap="Blues",

            xticklabels=["Predicted Control", "Predicted Cancer"],

            yticklabels=["Actual Control", "Actual Cancer"])

plt.title("LOOCV using Analyses and miRNA")

plt.xlabel("Predicted Labels")

plt.ylabel("Actual Labels")

plt.show()

 

# Calculate precision, recall/sensitivity, specificity, and F1-score

tn, fp, fn, tp = conf_matrix_loo.ravel()

 

precision_cancer = tp / (tp + fp)

precision_control = tn / (tn + fn)

 

recall_sensitivity_cancer = tp / (tp + fn)  # Recall/Sensitivity for Cancer

recall_sensitivity_control = tn / (tn + fp)  # Recall/Sensitivity for Control

 

specificity_cancer = tn / (tn + fp)  # Specificity for Cancer

specificity_control = recall_sensitivity_control  # Alias for clarity

 

f1_cancer = 2 * (precision_cancer * recall_sensitivity_cancer) / (precision_cancer + recall_sensitivity_cancer)

f1_control = 2 * (precision_control * recall_sensitivity_control) / (precision_control + recall_sensitivity_control)

 

support_cancer = sum(np.array(y) == "cancer")

support_control = sum(np.array(y) == "control")

 

# Create a PrettyTable for metrics

table = PrettyTable()

table.field_names = ["Class", "Precision", "Recall/Sensitivity", "Specificity", "F1-Score", "Support"]

 

# Add rows for each class

table.add_row(["Cancer", f"{precision_cancer:.2f}", f"{recall_sensitivity_cancer:.2f}", f"{specificity_cancer:.2f}",

               f"{f1_cancer:.2f}", support_cancer])

table.add_row(["Control", f"{precision_control:.2f}", f"{recall_sensitivity_control:.2f}", f"{specificity_control:.2f}",

               f"{f1_control:.2f}", support_control])

 

# Add macro and weighted averages

table.add_row(["Accuracy", "-", "-", "-", f"{accuracy_score(y, predictions):.2f}", len(y)])

table.add_row(["Macro Avg",

               f"{(precision_cancer + precision_control) / 2:.2f}",

               f"{(recall_sensitivity_cancer + recall_sensitivity_control) / 2:.2f}",

               f"{(specificity_cancer + specificity_control) / 2:.2f}",

               f"{(f1_cancer + f1_control) / 2:.2f}", len(y)])

table.add_row(["Weighted Avg",

               f"{precision_score(y, predictions, average='weighted'):.2f}",

               f"{recall_score(y, predictions, average='weighted'):.2f}",

               "-",

               f"{f1_score(y, predictions, average='weighted'):.2f}", len(y)])

 

# Print the pretty table

print("Metrics Summary:")

print(table)

 

# Compute ROC curve and AUC

fpr, tpr, _ = roc_curve(true_labels, y_probs)

roc_auc = auc(fpr, tpr)

 

# Plot ROC curve

plt.figure()

plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("Receiver Operating Characteristic")

plt.legend(loc="lower right")

plt.show()

 

# Calculate and plot feature importances after training the final model

feature_importances = clf_loo.feature_importances_

importances = pd.Series(feature_importances, index=X.columns) * 100  # Convert to percentages

 

# Sort and plot

sorted_importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))

sns.barplot(x=sorted_importances, y=sorted_importances.index)

plt.title("Feature Importance")

plt.xlabel("Importance Score (%)")

plt.ylabel("Features")

plt.show()