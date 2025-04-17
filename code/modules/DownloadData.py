# imports
import os
import pandas as pd
from joblib import Parallel, delayed
import logging

from code.utils import helpers as hlp


class DownloadData:

    def __init__(self, config, logger: logging.Logger):
        self.input_path = config['input_path']
        self.output_path = config['output_path']
        self.LOGGER = logger

    
    def download_data(self):
        """"
        Download both clinical and image data from the shared HPC area
        """

        self.LOGGER.info('Downloading metadata')
        self.metadata1 = self.download_metadata('dataset1', 'metadata1.csv')
        self.metadata2 = self.download_metadata('dataset2', 'metadata2.csv')

        self.LOGGER.info('Downloading clinical data')
        self.clinical1 = self.download_clinical_data('dataset1', 'clinical1.csv')
        self.clinical2 = self.download_clinical_data('dataset2', 'clinical2.csv')

        self.LOGGER.info('Downloading CT and segmented images')
        self.image_data_dict = self.download_images()



    def download_metadata(self, dataset: str, filename: str) -> pd.DataFrame:
        """
        Download the metadata for each dataset from the shared HPC area
        """

        # load the data
        input_filepath = os.path.join(self.input_path, dataset, 'metadata.csv')

        df = hlp.load_csv(input_filepath, self.LOGGER)

        # save the data
        if not os.path.exists(os.path.join(self.output_path, 'metadata')):
            os.makedirs(os.path.join(self.output_path, 'metadata'))

        output_filepath = os.path.join(self.output_path, 'metadata', filename)

        hlp.save_csv(df, input_filepath, output_filepath, self.LOGGER)

        return df

    
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


    def download_images(self) -> dict:
        """
        Download the CT images and segmented images from the shared HPC area
        """

        meta1 = self.metadata1
        meta1['Dataset'] = 'dataset1'
        meta2 = self.metadata2
        meta2['Dataset'] = 'dataset2'
        metadata = pd.concat([meta1, meta2], axis=0)

        self.joined_metadata = metadata.copy()

        seg_ids = metadata.loc[metadata['Modality']=='SEG', 'Subject ID'].unique().tolist()

        image_data_dict = {}
        image_dict_list = Parallel(n_jobs=-1)(delayed(self.get_images)(patient_id) for patient_id in seg_ids)

        for dictionary in image_dict_list:
            for patient_id, images in dictionary.items():
                image_data_dict[patient_id] = images

        output_directory = os.path.join(self.output_path, 'images')
        msg = (f"Downloaded images for {len(image_data_dict)} patients "
               f"and saved to {output_directory}")
        self.LOGGER.info(msg)
        
        return image_data_dict
    

    def get_images(self, patient_id: str) -> dict:
        """
        Function to allow loading and saving of each patients images
        to happen in parallel
        """

        image_data_dict = {}
        output_directory = os.path.join(self.output_path, 'images')

        seg_filepath = self.get_image_filepath(self.joined_metadata, patient_id, 'SEG')
        ct_filepath = self.get_image_filepath(self.joined_metadata, patient_id, 'CT')
        
        image_type_dict = {}
        
        # load the images
        loaded_seg = self.load_images(seg_filepath)
        loaded_ct = self.load_images(ct_filepath)
        image_type_dict['seg'] = [img for img, _ in loaded_seg]
        image_type_dict['ct'] = [img for img, _ in loaded_ct]

        # save the images as tif files
        seg_savepath = os.path.join(output_directory, patient_id, 'segmented')
        ct_savepath = os.path.join(output_directory, patient_id, 'ct')
        self.save_images(loaded_seg, seg_savepath, patient_id)
        self.save_images(loaded_ct, ct_savepath, patient_id)

        # populate the data dictionary
        image_data_dict[patient_id] = image_type_dict

        return image_data_dict


    def get_image_filepath(self, df: pd.DataFrame, patient_id: str, image_type: str) -> str:
        """
        Get the filepath to the image(s)
        """

        def access_dataframe_value(column: str) -> str:

            value = (
                df
                .loc[(df['Subject ID']==patient_id)&(df['Modality']==image_type),
                     column]
                .values[0]
            )
            return value

        dataset = access_dataframe_value('Dataset')
        filepath = access_dataframe_value('File Location')

        return os.path.join(self.input_path, dataset, filepath)
    

    def load_images(self, folderpath: str) -> list:
        """
        Load the images from the given filepath
        """

        image_filepaths = sorted(os.listdir(folderpath))
        images = []
        files = []
        for file in image_filepaths:
            filepath = os.path.join(folderpath, file)
            image = hlp.load_dicom(filepath, self.LOGGER)
            files.append(file)
            images.append(image)
        
        return list(zip(images, files))
    

    def save_images(self, images: list, folderpath: str, patient_id: str) -> None:
        """
        Save the images to the given filepath
        """

        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        
        for image, file in images:
            image_filename = file.replace('.dcm', '.tif')
            image_save_loc = os.path.join(folderpath, image_filename)
            hlp.save_medical_image(image, f"Image for {patient_id}",
                                   image_save_loc, self.LOGGER)
        return None
