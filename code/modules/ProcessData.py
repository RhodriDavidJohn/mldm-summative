# imports
import os
import pandas as pd
from configparser import ConfigParser
import logging
import sys

from code.utils import helpers as hlp


class ProcessData:
    def __init__(self, config: ConfigParser, logger: logging.Logger,
                 clinical1: pd.DataFrame, clinical2: pd.DataFrame,
                 seg_image_data: dict, ct_image_data: dict):
        
        self.output_path = config['output_path']
        self.LOGGER = logger
        self.clinical1 = clinical1
        self.clinical2 = clinical2
        self.seg_image_data = seg_image_data
        self.ct_image_data = ct_image_data

    
    def process_data(self) -> None:
        """
        Process the clinical data and (PET)CT images
        ready for training the models
        """
        
        self.LOGGER.info('Processing the first clinical data')
        self.clinical1_clean = self.process_single_clinical_data(self.clinical1)
        
        self.LOGGER.info('Processing the second clinical data')
        self.clinical2_clean = self.process_single_clinical_data(self.clinical2)
        
        self.LOGGER.info('Joining the processed clinical datasets')
        self.clinical_clean = self.process_clinical_data()
        
        self.LOGGER.info('Processing the CT images')
        self.image_features_df = self.process_images()

        return None

    
    def process_single_clinical_data(self, df) -> pd.DataFrame:
        """
        Process both clinical datasets individually.
        The processing steps are:


        Finally, the processed data are saved to the clean data folder.
        """

        pass

    
    def process_clinical_data(self) -> pd.DataFrame:
        """
        Join the processed clinical datasets together.
        The processing steps are:


        Finally, the processed data are saved to the clean data folder.
        """

        pass

    
    def process_images(self) -> pd.DataFrame:
        """
        Process the CT images with the segmented CT images to
        extract tumour properties for training the models.
        The processing steps are:


        Finally, the processed data are saved to the clean data folder.
        """

        pass