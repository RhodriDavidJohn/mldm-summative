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

        self.LOGGER.info('Downloading (PET)CT images')
        self.seg_image_data_dict = self.download_ct_data(segmented=False)

        self.LOGGER.info('Downloading (PET)CT segmented images')
        self.seg_image_data_dict = self.download_ct_data(segmented=True)



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


    def download_ct_data(self, segmented: bool) -> dict:
        """
        Download the (PET)CT image data from the shared HPC area
        """

        if segmented:
            folder_filter = 'Segmentation'
            image_filename = 'seg_ct_image.png'
            image_type = 'Segmented image'
        else:
            folder_filter = '1.00'
            image_filename = 'orig_ct_image.png'
            image_type = 'Image'

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
            for dir in os.listdir(subfolder):
                if folder_filter in dir:
                    folder = os.path.join(subfolder, dir)
                    break
                

            # read the image file
            assert len(os.listdir(folder))==1
            assert os.listdir(folder)[0].endswith('.dcm')

            file = os.path.join(folder, os.listdir(folder)[0])
            image = hlp.load_dicom(file, self.LOGGER)


            # save the image as jpeg
            patient_save_path = os.path.join(output_directory, patient_id)
            if not os.path.exists(patient_save_path):
                os.makedirs(patient_save_path)
            
            image_save_loc = os.path.join(patient_save_path, image_filename)
            hlp.save_medical_image(image, f"{image_type} for {patient_id}",
                                   image_save_loc, self.LOGGER)

            # populate the data dictionary
            image_data_dict[patient_id] = image
        
        msg = (f"Downloaded {len(image_data_dict)} {image_type.lower()}s "
               f"and saved to {output_directory}")
        self.LOGGER.info(msg)
        
        return image_data_dict

