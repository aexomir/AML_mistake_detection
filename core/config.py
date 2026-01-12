from argparse import ArgumentParser
import torch
from constants import Constants as const


def get_device():
    """Check if CUDA is available, return 'cuda' if available, 'cpu' otherwise."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.backbone = "omnivore"
        self.modality = "video"
        self.phase = "train"
        self.segment_length = 1

        # Use this for 1 sec video features
        self.segment_features_directory = "data/"

        self.ckpt_directory = "/data/rohith/captain_cook/checkpoints/"
        self.split = "recordings"
        self.batch_size = 1
        self.test_batch_size = 1
        self.num_epochs = 10
        self.lr = 1e-3
        self.weight_decay = 1e-3
        self.log_interval = 5
        self.dry_run = False
        self.ckpt = None
        self.seed = 1000
        # Device will be set after parsing args

        self.variant = const.TRANSFORMER_VARIANT
        self.model_name = None
        self.task_name = const.ERROR_RECOGNITION
        self.error_category = None

        self.enable_wandb = True

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.save_model = True
        self.__dict__.update(self.args)
        
        # Override device if explicitly provided via CLI, otherwise use auto-detected device
        if self.args.get('device') is not None:
            self.device = self.args['device']
        else:
            self.device = get_device()
        
        # Print device information
        print(f"Using device: {self.device}")
        if self.device == "cuda" and torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description="training code")

        # ----------------------------------------------------------------------------------------------
        # CONFIGURATION PARAMETERS
        # ----------------------------------------------------------------------------------------------

        parser.add_argument("--batch_size", type=int, default=1, help="batch size")
        parser.add_argument("--test-batch-size", type=int, default=1, help="input batch size for testing (default: 1000)")
        parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
        parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
        parser.add_argument("--seed", type=int, default=42, help="random seed (default: 1000)")

        parser.add_argument("--backbone", type=str, default=const.OMNIVORE, 
                            choices=[const.OMNIVORE, const.SLOWFAST, const.RESNET3D, const.X3D, const.IMAGEBIND, const.EGOVLP, const.PERCEPTIONENCODER], 
                            help="backbone model")
        parser.add_argument("--ckpt_directory", type=str, default="/data/rohith/captain_cook/checkpoints", help="checkpoint directory")
        parser.add_argument("--split", type=str, default=const.RECORDINGS_SPLIT, help="split")
        parser.add_argument("--variant", type=str, default=const.TRANSFORMER_VARIANT, help="variant")
        parser.add_argument("--model_name", type=str, default=None, help="model name")
        parser.add_argument("--task_name", type=str, default=const.ERROR_RECOGNITION, help="task name")
        parser.add_argument("--error_category", type=str, help="error category")
        parser.add_argument("--modality", type=str, nargs="+", default=[const.VIDEO], help="audio")
        parser.add_argument("--device", type=str, default=None, help="device to use (cuda/cpu). If not specified, auto-detects based on CUDA availability")
        
        # RNN-specific hyperparameters
        parser.add_argument("--rnn_hidden_size", type=int, default=256, help="RNN hidden size (default: 256)")
        parser.add_argument("--rnn_num_layers", type=int, default=2, help="Number of RNN layers (default: 2)")
        parser.add_argument("--rnn_dropout", type=float, default=0.2, help="RNN dropout rate (default: 0.2)")
        parser.add_argument("--rnn_bidirectional", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use bidirectional RNN (default: True)")
        parser.add_argument("--rnn_use_attention", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use attention pooling (default: False)")
        parser.add_argument("--rnn_type", type=str, default="LSTM", choices=["LSTM", "GRU"], help="RNN type: LSTM or GRU (default: LSTM)")

        return parser

    def set_model_name(self, model_name):
        self.model_name = model_name

    def print_config(self):
        """
        Prints the configuration
        :return:
        """
        print("Configuration:")
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")
        print("\n")
