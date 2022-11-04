from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=30, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.isTrain = False
        self.parser.add_argument("--has_real_image", action='store_true', help="if specified, display the real images on web") 
        self.parser.add_argument('--filter_empty', type=bool, default=False, help='filter empty patches with no labels')     
        self.parser.add_argument('--data_suffix', type=str, default='', help='dataset suffix for test')
        self.parser.add_argument('--use_focal_loss', action='store_true', help='use focal loss when training detection part')
        self.parser.add_argument('--focal_loss_gamma', type=float, default=5, help='hyperparameter: focal loss gamma')
        self.parser.add_argument('--mask_thresh', type=float, default=0.1, help='threshold of mask prob')
        self.parser.add_argument('--average_metric', action='store_true')
        self.parser.add_argument('--nuclei_thresh', type=int, default=3, help='threshold of nuclei size')
