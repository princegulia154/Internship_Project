import numpy as np
import pandas as pd
import tensorflow as tf
#from google.colab import drive
#drive.mount('/content/drive')

dftrain = pd.read_csv('https://raw.githubusercontent.com/princegulia154/Internship_Project/main/insurance_train.csv')
dfeval = pd.read_csv('https://raw.githubusercontent.com/princegulia154/Internship_Project/main/insurance_eval.csv')

# Convert 'smoker' column to integer (0 for non-smokers, 1 for smokers)
dftrain['smoker'] = dftrain['smoker'].map({'no': 0, 'yes': 1})
dfeval['smoker'] = dfeval['smoker'].map({'no': 0, 'yes': 1})

y_train = dftrain.pop('smoker')
y_eval = dfeval.pop('smoker')

categorical_column = ['sex', 'children', 'region']
numeric_column = ['age', 'bmi', 'charges']

feature_columns = []
for feature_name in categorical_column:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in numeric_column:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=2)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print("Accuracy:", result['accuracy'])

result = list(linear_est.predict(eval_input_fn))
insurance_id = int(input("Enter the insurance ID: "))
print(dfeval.loc[insurance_id])
print("Probability of Survival: ",result[insurance_id]['probabilities'][1])
if (y_eval.loc[insurance_id]==1):
  print("Smoker")
else:
  print("Non-Smoker")