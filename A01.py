import cv2
import sys
import numpy as np
from pathlib import Path

def slice_image(image, lower_slice, upper_slice):
    output = np.copy(image)
    output = np.where(output < lower_slice, 0, output)
    output = np.where(output > upper_slice, 0, output)
    return output

def main():
    argCount = len(sys.argv)
    if argCount < 5:
        print("ERROR: less than 5 parameters.")
        exit(1)
    
    filename = sys.argv[0]
    imagePath = sys.argv[1]
    lower_slice = int(sys.argv[2])
    upper_slice = int(sys.argv[3])
    outputDirectory = str(sys.argv[4])

    image = cv2.imread(imagePath, flags=cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("ERROR: Could not open the image.")
        exit(1)

    outputImage = slice_image(image, lower_slice, upper_slice)
    out_filename = "OUT_" + Path(filename).stem + "_" + str(lower_slice) + "_" + str(upper_slice) + ".png"
    
    resizedOutput = cv2.resize(outputImage, (1080, 720))
    cv2.imshow("Sliced Image", resizedOutput)

    fullOutputDirectory = outputDirectory + "/" + out_filename

    cv2.imwrite(fullOutputDirectory, resizedOutput)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()