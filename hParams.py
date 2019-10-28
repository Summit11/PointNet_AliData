class hParams:
    BATCH_SIZE = 1
    NUM_POINT = 58386
    VOXEL_POINT_COUNT = 15
    MAX_EPOCH = 50
    BASE_LEARNING_RATE = 0.001
    GPU_INDEX = 0
    MOMENTUM = 0.9
    OPTIMIZER = 'adam'
    DECAY_STEP = 200000
    DECAY_RATE = 0.7
    # VALID_VERBOSE = 10    #change

    EACH_FILE = 100
    NUM_CLASSES = 8
    TRAIN_FILE_NUM = 495  # change
    VALID_FILE_NUM = 5  # change
    TEST_FILE_NUM = 134

    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(DECAY_STEP)
    BN_DECAY_CLIP = 0.99

    train_csv_path = '/home/yue/D/阿里点云数据/dataset/training'
    test_csv_path = '/home/yue/D/阿里点云数据/TestSet'
    train_npy_path = '/home/yue/E/AliPointCloud/npy/train'
    test_npy_path = '/home/yue/E/AliPointCloud/npy/test'

    TRAIN_FILE_PATH = '/home/yue/D/data/AlibabaPointCloud/npy/train/'
    VALID_FILE_PATH = '/home/yue/D/data/AlibabaPointCloud/npy/valid/'

    RESULT_PATH = '/home/yue/桌面/Point/PointNetAli/results/'

    is_training = True