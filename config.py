##############################################################################
#                      Experiment parameters and settings                    #
##############################################################################
from algorithms.microsoft_ssa import microsoft_ssa
from algorithms.klcpd_ import KLCPD
from algorithms.bocpdms_ import BOCPDMS
from algorithms.hybrid_cusum_moving_window import hybrid_cusum_moving_window
from algorithms.hybrid_cusum import hybrid_cusum
from algorithms.binseg import binseg

from evaluation.classification import compute_f1_score

DATADIR = "data"

DATASETS = {
    "synth": [
        "struct",
        "mean",
        "energy"
        ],
    "yahoo": [
        "A4Benchmark-TS1",
        "A4Benchmark-TS2",
        "A4Benchmark-TS3", 
        "A4Benchmark-TS4", 
        "A4Benchmark-TS5",
        "A4Benchmark-TS6", 
        "A4Benchmark-TS7",
        "A4Benchmark-TS8", 
        "A4Benchmark-TS9",
        "A4Benchmark-TS10",
        "A4Benchmark-TS11",
        "A4Benchmark-TS12",
        "A4Benchmark-TS13",
        "A4Benchmark-TS14",
        "A4Benchmark-TS15",
        "A4Benchmark-TS16",
        "A4Benchmark-TS17",
        "A4Benchmark-TS18",
        "A4Benchmark-TS19",
        "A4Benchmark-TS20",
        "A4Benchmark-TS21",
        "A4Benchmark-TS22",
        "A4Benchmark-TS23",
        "A4Benchmark-TS24",
        "A4Benchmark-TS25",
        "A4Benchmark-TS26",
        "A4Benchmark-TS27",
        "A4Benchmark-TS28",
        "A4Benchmark-TS29",
        "A4Benchmark-TS30",
        "A4Benchmark-TS31",
        "A4Benchmark-TS32",
        "A4Benchmark-TS33",
        "A4Benchmark-TS34",
        "A4Benchmark-TS35",
        "A4Benchmark-TS36",
        "A4Benchmark-TS37",
        "A4Benchmark-TS38",
        "A4Benchmark-TS39",
        "A4Benchmark-TS40",
        "A4Benchmark-TS41",
        "A4Benchmark-TS42",
        "A4Benchmark-TS43",
        "A4Benchmark-TS44",
        "A4Benchmark-TS45",
        "A4Benchmark-TS46",
        "A4Benchmark-TS47",
        "A4Benchmark-TS48",
        "A4Benchmark-TS49",
        "A4Benchmark-TS50",
        "A4Benchmark-TS51",
        "A4Benchmark-TS52",
        "A4Benchmark-TS53",
        "A4Benchmark-TS54",
        "A4Benchmark-TS55",
        "A4Benchmark-TS56",
        "A4Benchmark-TS57",
        "A4Benchmark-TS58",
        "A4Benchmark-TS59",
        "A4Benchmark-TS60",
        "A4Benchmark-TS61",
        "A4Benchmark-TS62",
        "A4Benchmark-TS63",
        "A4Benchmark-TS64",
        "A4Benchmark-TS65",
        "A4Benchmark-TS66",
        "A4Benchmark-TS67",
        "A4Benchmark-TS68",
        "A4Benchmark-TS69",
        "A4Benchmark-TS70",
        "A4Benchmark-TS71",
        "A4Benchmark-TS72",
        "A4Benchmark-TS73",
        "A4Benchmark-TS74",
        "A4Benchmark-TS75",
        "A4Benchmark-TS76",
        "A4Benchmark-TS77",
        "A4Benchmark-TS78",
        "A4Benchmark-TS79",
        "A4Benchmark-TS80",
        "A4Benchmark-TS81",
        "A4Benchmark-TS82",
        "A4Benchmark-TS83",
        "A4Benchmark-TS84",
        "A4Benchmark-TS85",
        "A4Benchmark-TS86",
        "A4Benchmark-TS87",
        "A4Benchmark-TS88",
        "A4Benchmark-TS89",
        "A4Benchmark-TS90",
        "A4Benchmark-TS91",
        "A4Benchmark-TS92",
        "A4Benchmark-TS93",
        "A4Benchmark-TS94",
        "A4Benchmark-TS95",
        "A4Benchmark-TS96",
        "A4Benchmark-TS97",
        "A4Benchmark-TS98",
        "A4Benchmark-TS99",
    ]
    }

ALGORITHMS = {
    "microsoft_ssa":microsoft_ssa,
    "hybrid_cusum": hybrid_cusum, 
    "hybrid_cusum_moving_window": hybrid_cusum_moving_window,
    "binseg": binseg,
    "klcpd": KLCPD,
    "bocpdms": BOCPDMS
}



PARAMS = {
    "microsoft_ssa": {
        'training_window_size': [700, 400, 200, 50],
         'seasonal_window_size':[30, 15, 5],
         'change_history_length':[10], 
         'error_function': ['SignedDifference','AbsoluteDifference', 'SignedProportion', 'AbsoluteProportion', 'SquaredDifference'], 
         'martingale': ['Power', 'Mixture'], 
         'power_martingale_epsilon': [0.1], 
         'confidence': [95.0]
    },
    "hybrid_cusum": {
        'window_size': [700, 400, 200, 50], 
        'rows': [30, 15, 5], 
        'rank': [None], 
        'singular_threshold': [2, 5], 
        'distance_threshold': [10, 5], 
        'training_ratio': [0.5, 0.6], 
        'skip': [True, False]
    }, 
    "hybrid_cusum_moving_window": {
        'window_size': [700, 400, 200, 50], 
        'rows': [30, 15, 5], 
        'overlap_ratio': [0.0, 0.5, 0.9], 
        'rank': [None], 
        'singular_threshold': [2, 5], 
        'distance_threshold': [10, 5], 
        'training_ratio': [0.5, 0.6], 
        'skip': [True, False]
    }, 
    # "microsoft_ssa": {
    #     'training_window_size': [50, 10],
    #      'seasonal_window_size':[10],
    #      'change_history_length':[10], 
    #      'error_function': ['SignedDifference'], 
    #      'martingale': ['Power'], 
    #      'power_martingale_epsilon': [0.1], 
    #      'confidence': [95.0]
    # },
    "klcpd": {
        'lambda_real': [0.001, 0.1,1,10],
         'lambda_ae':[0.001,0.1,1,10],
         'wnd_dim':[5,10,20,25]
    },
    "bocpdms": {        
        "intensity": [50, 100, 200],
        "prior_a": [0.01, 1.0, 100],
        "prior_b": [0.01, 1.0, 100],
    },
    "binseg": {        
        "method": ["mean", "var", "meanvar"],
        "test_stat": ["Normal", "CUSUM", "CSS"],
        "max_cp": ["max", "default"],
        "penalty": [
            "None",
            "SIC",
            "BIC",
            "MBIC",
            "AIC",
            "Hannan-Quinn",
            "Asymptotic"]
    }
}


METRICS = {"compute_f1_score": compute_f1_score}

MARGIN = 10

RATIO = (0.3,0.3,0.4)
