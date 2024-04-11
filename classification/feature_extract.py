import SimpleITK as sitk
import os
import six
import csv
import pandas as pd

from radiomics import (
    glcm,
    glrlm,
    glszm,
    gldm,
    ngtdm,
    featureextractor
)
from radiomics import (
    firstorder,
    imageoperations,
    shape,
    shape2D,
)

def main():
    image_directory = r".\images"
    roi_directory = r".\masks"
    output = r".\output.csv"
    clinical_data_file = "ground_true.csv"

    # Read clinical data from the CSV file
    clinical_data = pd.read_csv(clinical_data_file)

    tmp_number = []
    settings = {}
    # settings["binWidth"] = 25
    # settings["Normalize"] = True
    # settings["resampledPixelSpacing"] = None
    # settings["interpolator"] = sitk.sitkBSpline
    # settings["label"] = 1

    for image_filename, roi_filename in zip(
        os.listdir(image_directory), os.listdir(roi_directory)
    ):
        image_path = os.path.join(image_directory, image_filename)
        roi_path = os.path.join(roi_directory, roi_filename)

        image = sitk.ReadImage(image_path, sitk.sitkInt8)
        mask = sitk.ReadImage(roi_path, sitk.sitkInt8)

        ndImg = sitk.GetArrayFromImage(image)

        bb, correctedMask = imageoperations.checkMask(image, mask)
        if correctedMask is not None:
            mask = correctedMask
        image, mask = imageoperations.cropToTumorMask(image, mask, bb)

        ############
        tmp = []
        tmp_name = []

        # Extract image features
        glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
        glcmFeatures.enableAllFeatures()
        results = glcmFeatures.execute()
        for key, val in six.iteritems(results):
            tmp_name.append(key)
            tmp.append(val)
        glrlmFeatures = glrlm.RadiomicsGLRLM(image, mask, **settings)
        glrlmFeatures.enableAllFeatures()

        results = glrlmFeatures.execute()

        for key, val in six.iteritems(results):
            tmp_name.append(key)
            tmp.append(val)
        glszmFeatures = glszm.RadiomicsGLSZM(image, mask, **settings)
        glszmFeatures.enableAllFeatures()

        results = glszmFeatures.execute()
        for key, val in six.iteritems(results):
            tmp_name.append(key)
            tmp.append(val)
        gldmFeatures = gldm.RadiomicsGLDM(image, mask, **settings)
        gldmFeatures.enableAllFeatures()

        results = gldmFeatures.execute()
        for key, val in six.iteritems(results):
            tmp_name.append(key)
            tmp.append(val)
        ngtdmFeatures = ngtdm.RadiomicsNGTDM(image, mask, **settings)
        ngtdmFeatures.enableAllFeatures()

        results = ngtdmFeatures.execute()

        for key, val in six.iteritems(results):
            tmp_name.append(key)
            tmp.append(val)
        tmp_name.append("name")
        tmp.append(image_filename)
        tmp_number.append(tmp)
    
    # Combine clinical data with image features
    combined_data = pd.concat([clinical_data, pd.DataFrame(tmp_number, columns=tmp_name)], axis=1)
    
    # Save the combined data to a CSV file
    combined_data.to_csv(output, index=False)

if __name__ == "__main__":
    main()
