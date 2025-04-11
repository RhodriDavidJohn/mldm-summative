# functions to download the clinical and
# (PET)CT scan data from shared HPC area


# imports
import os
import pandas as pd
from skimage import io
import logging
import sys

from code.utils import helpers as hlp


class DownloadData:

    def __init__(self, config, logger: logging.Logger):
        self.input_path = config['input_path']
        self.output_path = config['output_path']
        self.LOGGER = logger

    
    def download_data(self):
        """"
        Download both clinical and (PET)CT data from the shared HPC area
        """

        self.LOGGER.info('Downloading clinical data')
        self.clinical1 = self.download_clinical_data('dataset1', 'clinical1.csv')
        self.clinical2 = self.download_clinical_data('dataset2', 'clinical2.csv')

        self.LOGGER.info('Downloading (PET)CT segmented images data')
        self.seg_image_data_dict = self.download_segmented_ct_data()



    def download_clinical_data(self, dataset: str, filename: str) -> pd.DataFrame:
        """
        Download the clinical csv data from the shared HPC area
        """

        # load the data
        input_filepath = os.path.join(self.input_path, dataset, filename)

        df = hlp.load_csv(input_filepath, self.LOGGER)

        # save the data
        if not os.path.exists(os.path.join(self.output_path, 'clinical')):
            os.makedirs(os.path.join(self.output_path, 'clinical'))

        output_filepath = os.path.join(self.output_path, 'clinical', filename)

        hlp.save_csv(df, input_filepath, output_filepath, self.LOGGER)

        return df


    def download_segmented_ct_data(self):
        """
        Download the (PET)CT segmented image data from the shared HPC area
        """

        image_data_dict = {}

        input_filepath = os.path.join(self.input_path, 'dataset1')
        output_directory = os.path.join(self.output_path, 'ct_image')

        for patient_id in os.listdir(input_filepath):
            patient_folder = os.path.join(input_filepath, patient_id)

            # check if the item is a folder because
            # files exist in the directory
            if not os.path.isdir(patient_folder):
                continue
            
            # the patient folder should only contain one item
            # which is a subfolder
            try:
                assert len(os.listdir(patient_folder))==1
            except Exception as e:
                msg = (f"{patient_id} has {len(os.listdir(patient_folder))} "
                       f"subfolders instead of 1. The following error was raised: {e}")
                self.LOGGER.error(msg)
                sys.exit()

            subfolder = os.path.join(patient_folder, os.listdir(patient_folder)[0])

            assert os.path.isdir(subfolder)

            # the subfolder should contain 3 more folders
            # but we only want the one with the segmented image
            for folder in os.listdir(subfolder):
                if 'Segmentation' in folder:
                    seg_folder = os.path.join(subfolder, folder)
                    break
                

            # read the image file
            assert len(os.listdir(seg_folder))==1
            assert os.listdir(seg_folder)[0].endswith('.dcm')

            seg_file = os.path.join(seg_folder, os.listdir(seg_folder)[0])
            seg_image = hlp.load_dicom(seg_file, self.LOGGER)


            # save the image as jpeg
            patient_save_path = os.path.join(output_directory, patient_id)
            if not os.poath.exists(patient_save_path):
                os.makedirs(patient_save_path)
            image_filename =  f"seg_image_{patient_id}.jpg"
            image_save_loc = os.path.join(patient_save_path, image_filename)
            hlp.save_medical_image(seg_image, f"Segmented image for {patient_id}",
                                   image_save_loc, self.LOGGER)

            # populate the data dictionary
            image_data_dict[patient_id] = seg_image
        
        msg = (f"Downloaded {len(image_data_dict)} segemented images "
               f"and saved to {output_directory}")
        self.LOGGER.info(msg)
        
        return image_data_dict

