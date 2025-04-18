# main pipeline that gets run via the main.py file

import os
import pandas as pd
from skimage import io
from configparser import ConfigParser
import time
import sys

from code.utils import helpers as hlp

from code.modules.DownloadData import DownloadData
from code.modules.ProcessData import ProcessData


class Pipeline:

    def __init__(self, config: ConfigParser):
        self.config = config
        # set up the logger
        logs_folder = config.get('global', 'logs_folder')
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        self.run_id = hlp.create_run_id(logs_folder)
        self.LOGGER = hlp.setup_logger(logs_folder, self.run_id)


    def run(self):
        
        # start timer
        start_time = time.time()

        self.LOGGER.info("Starting pipeline run")
        self.LOGGER.info(f"RUN_ID: {self.run_id}")

        global_config = self.config['global']

        try:
            assert global_config['run_modular'] in ['True', 'False']
        except Exception as e:
            self.LOGGER.error("Config option 'run_modular' must be set to either True or False.")
            raise(e)

        if global_config['run_modular']=='True':
            # run the modlar pipeline
            self.modular_pipeline(module=global_config['module'])
        else:
            # run the main pipeline
            self.main_pipeline()

        # end the timer
        end_time = time.time()
        hours, remainder = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        self.LOGGER.critical(f'Pipeline finished successfully in {hours} hrs {minutes} mins {seconds} secs')


    def main_pipeline(self):
        """"
        Main pipeline which runs each module sequentially.
        The modules run in the following order:
        1. DownloadData
        2. ProcessData
        3. TrainModels
        4. EvaluateModels
        """

        self.LOGGER.info('Running the main pipeline')

        # download the data
        dd = DownloadData(config=self.config['download_data'], logger=self.LOGGER)
        dd.download_data()

        # process the data
        processer = ProcessData(config=self.config['process_data'], logger=self.LOGGER,
                                clinical1=dd.clinical1, clinical2=dd.clinical2,
                                image_data=dd.image_data_dict)
        processer.process_data()

        # train the models


        # evaluate the models

    

    def modular_pipeline(self, module):
        """
        Modular pipeline which runs each module independently.
        """

        self.LOGGER.info('Running the modular pipeline')

        valid_modules = ['download', 'process', 'train', 'evaluate']

        try:
            assert module in valid_modules
        except AssertionError:
            msg = (
                'Module specified in the modular config file is not valid. '
                f'The specified module is: {module}. '
                f'Valid modules are: {valid_modules}'
            )
            self.LOGGER.error(msg)
            sys.exit()
        
        # run specified module
        if module == 'download':
            self.LOGGER.info('Running the data download module')
            dd = DownloadData(config=self.config['download_data'], logger=self.LOGGER)
            dd.download_data()

        elif module == 'process':
            self.LOGGER.info('Running the data processing module')
            processer_config = self.config['process_data']
            # load the data into a dictionary
            clinical1 = pd.read_csv(processer_config['clinical1_path'])
            clinical2 = pd.read_csv(processer_config['clinical2_path'])
            image_inputs = processer_config['image_dir']
            image_data = {}
            for patient_id in sorted(os.listdir(image_inputs)):
                ct_image_path = os.path.join(image_inputs, patient_id, 'ct')
                seg_image_path = os.path.join(image_inputs, patient_id, 'segmented')
                # load the images
                ct_images = []
                for file in sorted(os.listdir(ct_image_path)):
                    ct_image_filepath = os.path.join(ct_image_path, file)
                    ct_img = io.imread(ct_image_filepath)
                    ct_images.append(ct_img)
                seg_image = []
                for file in sorted(os.listdir(seg_image_path)):
                    seg_image_filepath = os.path.join(seg_image_path, file)
                    seg_image.append(io.imread(seg_image_filepath))

                image_data[patient_id] = {
                    'ct': ct_images,
                    'segmented': seg_image
                }

            # run the module
            processer = ProcessData(config=processer_config, logger=self.LOGGER,
                                    clinical1=clinical1, clinical2=clinical2,
                                    image_data=image_data)
            processer.process_data()

        elif module == 'train':
            self.LOGGER.info('Running the model training module')
            pass

        elif module == 'evaluate':
            self.LOGGER.info('Running the model evaluation module')
            pass
