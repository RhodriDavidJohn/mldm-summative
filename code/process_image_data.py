import pandas as pd
from sys import argv

from utils.image_processing import *
from utils import helpers2 as hlp



def process_images(batch: list, save_loc: str) -> None:

    metadata = hlp.get_metadata()

    data = []

    for patient_id in batch:
        
        # load the data
        seg_folderpath = get_image_filepath(metadata, patient_id, 'SEG')
        ct_folderpath = get_image_filepath(metadata, patient_id, 'CT')

        image_list_dict = load_images(patient_id, seg_folderpath, ct_folderpath)

        image_dict = {
            'seg': image_list_dict['seg'][0],
            'ct': create_3d_image(image_list_dict['ct'])
        }

        del image_list_dict

        # process the data
        try:
            features = extract_tumour_properties(img=image_dict['ct'],
                                                 mask=image_dict['seg'])
        except Exception as e:
            print(f"Error processing images for patient ID {patient_id}: {e}")
            raise(e)
        data.append([patient_id] + features)
    
    columns = (["patient_id"] +
               ["n_tumours", "maximum_diameter", "surface_area", # segmented features
                "surface_to_volume_ratio", "volume", "radius"] +
               ["mean_intensity", "std_intensity", "min_intensity", # intensity features
                "max_intensity", "median_intensity"] +
               ["contrast", "correlation", "dissimilarity", "homogeneity"]) # grayscale features

    df = pd.DataFrame(data=data, columns=columns)

    clinical = hlp.load_csv('data/clean/clinical_joined.csv')

    clinical = clinical[['patient_id', 'death_2years']]

    df = df.merge(right=clinical, on='patient_id', how='left')

    df = df.dropna(subset=["death_2years"])

    hlp.save_csv(df, 'image features', save_loc)

    return None


if __name__=="__main__":
    batch_num = int(argv[1])
    batch = argv[2].split(',')

    save_loc = f"data/clean/image_features_{batch_num}.csv"

    process_images(batch, save_loc)