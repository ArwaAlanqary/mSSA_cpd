##############################################################################
#                      Experiment parameters and settings                    #
##############################################################################

DATASETS = {
    "frequency": [ 
        "struct0", "struct1", "struct2", "struct3", "struct4", "struct5", 
        "struct6", "struct7", "struct8", "struct9", "struct10", "struct11",
        "struct12", "struct13", "struct14", "struct15", "struct16", 
        "struct17", "struct18", "struct19"
    ], 
    "mean": [ 
        "mean0", "mean1", "mean2", "mean3", "mean4", "mean5", "mean6", "mean7", 
        "mean8", "mean9", "mean10", "mean11", "mean12", 
        "mean13", "mean14", "mean15","mean16", "mean17", "mean18" , "mean19"
    ],
    "energy": [ 
        "energy0", "energy1", "energy2", "energy3", "energy4", "energy5", "energy6",
        "energy7", "energy8", "energy9", "energy10", "energy11", "energy12", "energy13",
        "energy14", "energy15", "energy16", "energy17", "energy18", "energy19" 
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
    ], 
    'hasc': [
        'HASC1001', 
        'HASC1002', 
        'HASC1003', 
        'HASC1004',
        'HASC1005',
        'HASC1006',
        'HASC1007',
        'HASC1008',
        'HASC1009',
        'HASC1010',
        'HASC1011',
        'HASC1012',
        'HASC1013',
        'HASC1014',
        'HASC1015',
        'HASC1016',
        'HASC1017',
        'HASC1018',
    ],
    "beedance": [ 
        "beedance-1",
        "beedance-2",
        "beedance-3",
        "beedance-4",
        "beedance-5",
        "beedance-6"],
    "occupancy": [ 
        "occupancy"],
    "mixed": [ 
        "multi0",
        "multi1",
        "multi2",
        "multi3",
        "multi4",]}

PARAMS = {
    "microsoft_ssa": {
        'training_window_size': [0.6], #* mean interval length between change points (across dataset)
         'seasonal_window_size':[1, 0.5, 0.1], # sqrt(training_window_size)
         'change_history_length':[10, 30, 50],
         'error_function': ['SignedDifference','AbsoluteDifference', 'SignedProportion', 'AbsoluteProportion', 'SquaredDifference'], 
         'martingale': ['Power', 'Mixture'], 
         'power_martingale_epsilon': [0.1, 0.5], 
         'confidence': [95.0]
    },
    "mssa": {
        'window_size': [0.6], #* mean interval length between change points (across dataset)
        'rows': [1, 0.5, 0.1], # * sqrt(training_window_size)
        'rank': [0.5, 0.95, 3, 5], 
        'distance_threshold': [1, 5, 10], 
        'training_ratio': [0.9], 
        'skip': [False],
        'normalize': [True]
    }, 
    "mssa_mw": {
        'window_size': [0.6], #* mean interval length between change points (across dataset)
        'rows':  [1, 0.5, 0.1], # * sqrt(training_window_size)
        'rank': [0.5, 0.95, 3, 5], 
        'distance_threshold': [1, 5, 10], 
        'training_ratio': [0.9], 
        'skip': [False],
        'normalize': [True]
    }, 
    "klcpd": {
        'lambda_real': [0.001, 0.1,1,10],
         'lambda_ae':[0.001,0.1,1,10],
         'wnd_dim':[25]
    },
    "bocpdms": {        
        "intensity": [50, 100, 200],
        "prior_a": [0.01, 1.0, 100],
        "prior_b": [0.01, 1.0, 100],
    },
    "binseg": {        
        "method": ["mean", "var", "meanvar"],
        "test_stat": ["Normal", "CUSUM", "CSS"],
        "max_cp": ["max", "default"], #Max = total number of cp in the dataset # default = total number of cp/number of ts
        "penalty": [
            "None",
            "SIC",
            "BIC",
            "MBIC",
            "AIC",
            "Hannan-Quinn",
            "Asymptotic"] # 0.05
    },
    "no_change": {
    
    }
}

DEFAULT = {
    "microsoft_ssa": {
        'change_history_length':10,
        'confidence': 95.0,
        'error_function': 'SignedDifference',
        'martingale': 'Power', 
        'power_martingale_epsilon': 0.1,
        'seasonal_window_size': 1, 
        'training_window_size': 0.6, 
    },
    "mssa": {
        'distance_threshold': 5, 
        'normalize': True,
        'rank': 0.95, 
        'rows': 1,
        'skip': False,
        'training_ratio': 0.9,
        'window_size': 0.6
    }, 
    "mssa_mw": {
        'distance_threshold': 5, 
        'normalize': True,
        'rank': 0.95, 
        'rows': 1,
        'skip': False,
        'training_ratio': 0.9,
        'window_size': 0.6
    },
    "klcpd": {
        'lambda_ae': 0.001, 
        'lambda_real': 0.1, 
        'wnd_dim': 25
    },

    "bocpdms": {        
        "intensity": 100,
        "prior_a": 1.0,
        "prior_b": 1.0,
    },

    "binseg": {
        'max_cp': 'default', 
        'method': 'mean', 
        'penalty': 'MBIC', 
        'test_stat': 'Normal'
    },

    "no_change": {
    
    }
}


MARGIN = 10

