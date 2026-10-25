import numpy as np

# Radio/energy model 

E_ELEC = 50e-9          
E_FS   = 10e-12         
E_MP   = 0.0013e-12    
E_DA   = 5e-9         
D0     = 87.0
PACKET_BITS = 4000

# Network geometry  
AREA_X = 100.0
AREA_Y = 100.0

INIT_ENERGY = 0.5        

MAX_ROUNDS = 60000 

# ESCA
ESCA_KMAX_FRACTION = 0.20   

# IGHND
GRID_M = 4  

ANP_WEIGHTS = dict(W1=0.4, W2=0.36, W3=0.14, W4=0.1) 

ADV_ALPHA = 1.0
ADV_BETA = 1.0

WMS_D = 0.5
WMS_SIGMA = 0.5

LOWER_BOUND_FACTOR = 0.5
UPPER_BOUND_FACTOR = 1.5

MNODE_NATIVE = 1.0
MNODE_MERGED = 0.5

#dataset
DATASET_PATH = "data/converted_dataset.csv"


DATASET_MAX_NODES = 300
DATASET_SAMPLE_SEED = 0   
DATASET_SCENARIOS = {
    "DATASET-CenterBS": "centroid",
    "DATASET-CornerBS": "corner",
}


FIXED_SEEDS = [42, 101, 202, 303, 404]
N_TRIALS = len(FIXED_SEEDS)


CURVE_SEED = FIXED_SEEDS[0]

# GraphSAGE 
GRAPHSAGE_KNN_K        = 6      
GRAPHSAGE_HIDDEN1      = 8
GRAPHSAGE_HIDDEN2      = 8
GRAPHSAGE_EPOCHS       = 20   
GRAPHSAGE_LR           = 0.05
GRAPHSAGE_BLEND_ALPHA  = 0.5    
GRAPHSAGE_SEED         = 0


PACKET_LOSS_ALPHA = 0.02   
