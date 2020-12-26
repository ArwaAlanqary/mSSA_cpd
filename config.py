##############################################################################
#                      Experiment parameters and settings                    #
##############################################################################
from algorithms.microsoft_ssa import microsoft_ssa
from algorithms.klcpd_ import KLCPD

DATADIR = "data"

DATASETS = [
    "synth/struct_ts",
    "synth/mean_ts",
    "synth/energy_ts",
    "yahoo/A4Benchmark-TS1_ts",
    "yahoo/A4Benchmark-TS2_ts",
    "yahoo/A4Benchmark-TS3_ts", 
    "yahoo/A4Benchmark-TS4_ts", 
    "yahoo/A4Benchmark-TS5_ts",
    "yahoo/A4Benchmark-TS6_ts", 
    "yahoo/A4Benchmark-TS7_ts",
    "yahoo/A4Benchmark-TS8_ts", 
    "yahoo/A4Benchmark-TS9_ts",
    "yahoo/A4Benchmark-TS10_ts",
    "yahoo/A4Benchmark-TS11_ts",
    "yahoo/A4Benchmark-TS12_ts",
    "yahoo/A4Benchmark-TS13_ts",
    "yahoo/A4Benchmark-TS14_ts",
    "yahoo/A4Benchmark-TS15_ts",
    "yahoo/A4Benchmark-TS16_ts",
    "yahoo/A4Benchmark-TS17_ts",
    "yahoo/A4Benchmark-TS18_ts",
    "yahoo/A4Benchmark-TS19_ts",
    "yahoo/A4Benchmark-TS20_ts",
    "yahoo/A4Benchmark-TS21_ts",
    "yahoo/A4Benchmark-TS22_ts",
    "yahoo/A4Benchmark-TS23_ts",
    "yahoo/A4Benchmark-TS24_ts",
    "yahoo/A4Benchmark-TS25_ts",
    "yahoo/A4Benchmark-TS26_ts",
    "yahoo/A4Benchmark-TS27_ts",
    "yahoo/A4Benchmark-TS28_ts",
    "yahoo/A4Benchmark-TS29_ts",
    "yahoo/A4Benchmark-TS30_ts",
    "yahoo/A4Benchmark-TS31_ts",
    "yahoo/A4Benchmark-TS32_ts",
    "yahoo/A4Benchmark-TS33_ts",
    "yahoo/A4Benchmark-TS34_ts",
    "yahoo/A4Benchmark-TS35_ts",
    "yahoo/A4Benchmark-TS36_ts",
    "yahoo/A4Benchmark-TS37_ts",
    "yahoo/A4Benchmark-TS38_ts",
    "yahoo/A4Benchmark-TS39_ts",
    "yahoo/A4Benchmark-TS40_ts",
    "yahoo/A4Benchmark-TS41_ts",
    "yahoo/A4Benchmark-TS42_ts",
    "yahoo/A4Benchmark-TS43_ts",
    "yahoo/A4Benchmark-TS44_ts",
    "yahoo/A4Benchmark-TS45_ts",
    "yahoo/A4Benchmark-TS46_ts",
    "yahoo/A4Benchmark-TS47_ts",
    "yahoo/A4Benchmark-TS48_ts",
    "yahoo/A4Benchmark-TS49_ts",
    "yahoo/A4Benchmark-TS50_ts",
    "yahoo/A4Benchmark-TS51_ts",
    "yahoo/A4Benchmark-TS52_ts",
    "yahoo/A4Benchmark-TS53_ts",
    "yahoo/A4Benchmark-TS54_ts",
    "yahoo/A4Benchmark-TS55_ts",
    "yahoo/A4Benchmark-TS56_ts",
    "yahoo/A4Benchmark-TS57_ts",
    "yahoo/A4Benchmark-TS58_ts",
    "yahoo/A4Benchmark-TS59_ts",
    "yahoo/A4Benchmark-TS60_ts",
    "yahoo/A4Benchmark-TS61_ts",
    "yahoo/A4Benchmark-TS62_ts",
    "yahoo/A4Benchmark-TS63_ts",
    "yahoo/A4Benchmark-TS64_ts",
    "yahoo/A4Benchmark-TS65_ts",
    "yahoo/A4Benchmark-TS66_ts",
    "yahoo/A4Benchmark-TS67_ts",
    "yahoo/A4Benchmark-TS68_ts",
    "yahoo/A4Benchmark-TS69_ts",
    "yahoo/A4Benchmark-TS70_ts",
    "yahoo/A4Benchmark-TS71_ts",
    "yahoo/A4Benchmark-TS72_ts",
    "yahoo/A4Benchmark-TS73_ts",
    "yahoo/A4Benchmark-TS74_ts",
    "yahoo/A4Benchmark-TS75_ts",
    "yahoo/A4Benchmark-TS76_ts",
    "yahoo/A4Benchmark-TS77_ts",
    "yahoo/A4Benchmark-TS78_ts",
    "yahoo/A4Benchmark-TS79_ts",
    "yahoo/A4Benchmark-TS80_ts",
    "yahoo/A4Benchmark-TS81_ts",
    "yahoo/A4Benchmark-TS82_ts",
    "yahoo/A4Benchmark-TS83_ts",
    "yahoo/A4Benchmark-TS84_ts",
    "yahoo/A4Benchmark-TS85_ts",
    "yahoo/A4Benchmark-TS86_ts",
    "yahoo/A4Benchmark-TS87_ts",
    "yahoo/A4Benchmark-TS88_ts",
    "yahoo/A4Benchmark-TS89_ts",
    "yahoo/A4Benchmark-TS90_ts",
    "yahoo/A4Benchmark-TS91_ts",
    "yahoo/A4Benchmark-TS92_ts",
    "yahoo/A4Benchmark-TS93_ts",
    "yahoo/A4Benchmark-TS94_ts",
    "yahoo/A4Benchmark-TS95_ts",
    "yahoo/A4Benchmark-TS96_ts",
    "yahoo/A4Benchmark-TS97_ts",
    "yahoo/A4Benchmark-TS98_ts",
    "yahoo/A4Benchmark-TS99_ts",
    
]

ALGORITHMS = {
    "microsoft_ssa":microsoft_ssa,
    "hybrid_cusum": KLCPD, 
    "hybrid_cusum_moving_window": KLCPD,
    "klcpd": KLCPD
}



PARAMS = {
    "microsoft_ssa": {
        'training_window_size': [50, 30],
         'seasonal_window_size':[10,5],
         'change_history_length':[8, 10], 
         'error_function': ['SignedDifference','AbsoluteDifference', 'SignedProportion', 'AbsoluteProportion', 'SquaredDifference'], 
         'martingale': ['Power', 'Mixture'], 
         'power_martingale_epsilon': [0.1], 
         'confidence': [95.0], 
         'columns': [{'result': 'ts'}]
    }
        "klcpd": {
        'lambda_real': [0.001, 0.1,1,10],
         'lambda_ae':[0.001,0.1,1,10],
         'wnd_dim':[5,10,20,25]
    }
}


METRICS = {"compute_f1_score"}

MARGIN = 10
