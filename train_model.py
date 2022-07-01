import os
import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

def main():
    ### Prepare the data ###
    # load data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # use DMatrix for xgbosot
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    ### Training ###
    # set xgboost params
    param = {
        'max_depth': 3,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': 3}  # the number of classes that exist in this datset
    num_round = 20  # the number of training iterations

    # training and testing - numpy matrices
    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)

    # extracting most confident predictions
    best_preds = np.asarray([np.argmax(line) for line in preds])
    print("Numpy array precision:", precision_score(y_test, best_preds, average='macro'))

    ### Save model ###
    # Create the model repository directory
    REPO_PATH = 'model_repository'
    os.makedirs(REPO_PATH, exist_ok=True)
    model_dir = os.path.join(REPO_PATH, 'iris_xgboost')
    version_dir = os.path.join(model_dir, '1')
    os.makedirs(version_dir, exist_ok=True)
    model_file = os.path.join(version_dir, 'xgboost.json')
    bst.save_model(model_file)

    # parameter to passed into config text
    MAX_MEMORY_BYTES = 60_000_000
    features = X_test.shape[1]
    num_classes = np.unique(y_test).size
    bytes_per_sample = (features + num_classes) * 4
    max_batch_size = MAX_MEMORY_BYTES // bytes_per_sample

    config_text = f"""backend: "fil"
max_batch_size: {max_batch_size}
input [                                 
{{  
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {features} ]                    
}} 
]
output [
{{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {num_classes} ]
}}
]
instance_group [{{ kind: 'KIND_GPU' }}]
parameters [
{{
    key: "model_type"
    value: {{ string_value: "xgboost_json" }}
}},
{{
    key: "predict_proba"
    value: {{ string_value: "true" }}
}},
{{
    key: "output_class"
    value: {{ string_value: "true" }}
}},
{{
    key: "threshold"
    value: {{ string_value: "0.5" }}
}},
{{
    key: "storage_type"
    value: {{ string_value: "AUTO" }}
}}
]

dynamic_batching {{
max_queue_delay_microseconds: 100
}}"""
    config_path = os.path.join(model_dir, 'config.pbtxt')
    with open(config_path, 'w') as file_:
        file_.write(config_text)

if __name__ == "__main__":
    main()