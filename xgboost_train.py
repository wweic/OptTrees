import sys
import time

import xgboost as xgb

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]

    # read in data
    dtrain = xgb.DMatrix(train_data)
    dtest = xgb.DMatrix(test_data)

    # specify parameters via map
    param = {'max_depth':2, 'eta':1, 'silent':1 }
    num_round = 2

    print("Training")
    bst = xgb.train(param, dtrain, num_round)
    bst.dump_model('/tmp/tree.model')

    print("Predicting")
    start = time.time()
    preds = bst.predict(dtest)    
    total = time.time() - start
    print("{} seconds {}".format(total, preds))

if __name__ == '__main__':
    main()