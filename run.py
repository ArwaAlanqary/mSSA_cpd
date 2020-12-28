from config import DATADIR, DATASETS, ALGORITHMS, PARAMS, METRICS, MARGIN

from optimizing.grid_search import grid_search
from evaluation.classification import compute_f1_score

#import models
from algorithms.microsoft_ssa import microsoft_ssa
from algorithms.klcpd_ import KLCPD



# choose one algorithm for now
## otherwise do for algorithm_name, algorithm in ALGORITHMS.items()
algorithm = KLCPD
algorithm_name = 'klcpd'
# data
# choose one for now
# otherwise do for data in DATASETS
dataset = "synth/struct_ts",
data_path = os.path.join(os.getcwd(), "Data")
data = pd.read_csv(os.path.join(data_path,  "{}_ts.csv".format(dataset)), header=None)
labels = pd.read_csv(os.path.join(data_path, "{}_labels.csv".format(dataset)), header=None)

######### SPLIT? #############
ts = data[1].values
# ts_train, labels_train
# ts_validate, labels_validate
# ts_test, labels_test





optimizer = grid_search(param, microsoft_ssa, compute_f1_score)
optimizer.search(ts, labels, margin)

## Generate model results with best parameteres
model = microsoft_ssa(**optimizer.best_param)
model.train()
model.detect(ts)

score = compute_f1_score(labels, model.cp, MARGIN)

# store algorithm/data/f1_score 
with open('f1_scores.csv','a+') as outfile:
    outfile.write(f'{algorithm_name},{dataset},{score},{MARGIN} \n')