# main pipeline that gets run via the main.py file

import os
import pandas as pd
from configparser import ConfigParser
import time
import sys

from code.utils import helpers as hlp

from code.modules.DownloadData import DownloadData


class Pipeline:

    def __init__(self, config: ConfigParser):
        self.config = config
        # set up the logger
        logs_folder = config.get('global', 'logs_folder')
        self.run_id = hlp.create_run_id(logs_folder)
        self.LOGGER = hlp.setup_logger(logs_folder, self.run_id)


    def run(self):
        
        # start timer
        start_time = time.time()

        self.LOGGER.info("Starting pipeline run")
        self.LOGGER.info(f"RUN_ID: {self.run_id}")

        global_config = self.config['global']

        if bool(global_config['run_modular']):
            # run the modlar pipeline
            self.LOGGER.info('Running the modular pipeline')
            self.modular_pipeline(module=global_config['module'])
        else:
            # run the main pipeline
            self.LOGGER.info('Running the main pipeline')
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

        # download the data
        dd = DownloadData(config=self.config['data_download'], logger=self.LOGGER)
        dd.download_data()

        # process the data


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
            pass

        elif module == 'train':
            self.LOGGER.info('Running the model training module')
            pass

        elif module == 'evaluate':
            self.LOGGER.info('Running the model evaluation module')
            pass
