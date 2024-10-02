import argparse
import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseOptions:
    """
    This class defines options used during all types of experiments.
    """
    def __init__(self) -> None:
        """
        Initializes the BaseOptions class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self) -> None:
        self.parser.add_argument(
            "--experiment_name",
            type=str,
            default="Semiconductor Manufacturing Unit (SECOM) Failure Prediction",
            help="Name of the experiment",
        ),

        self.parser.add_argument(
            "--data_path",
            type=str,
            default="/Users/karthik/Desktop/Datasets/SECOM-Dataset/uci-secom.csv",
            help="Path to the data file",
        )

        self.parser.add_argument(
            "--wrapper_width",
            type=int,
            default=50,
            help="Width of the wrapper",
        )
        
        self.parser.add_argument(
            "--saved_model_path",
            type=str,
            default="/Users/karthik/SECOM-Dissertation-CODE/artifacts/models/",
            help="Path to save the trained model",
        )
        
        self.parser.add_argument(
            "--log_path",
            type=str,
            default="/Users/karthik/SECOM-Dissertation-CODE/artifacts/logs/",
            help="Path to save the logs",
        )
        
        self.initialized = True

    def parse(self):
        """
        Parses the arguments passed to the script

        Parameters
        ----------
        None

        Returns
        -------
        opt: argparse.Namespace
            The parsed arguments
        """
        if not self.initialized:
            self.initialize()
        self._opt = self.parser.parse_args()

        return self._opt
