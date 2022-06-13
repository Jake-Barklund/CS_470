import cv2
import sys
import numpy as np

def applyFilter(image, kernel):
    fimage = image.astype("float64")
    fkernel = kernel.astype("float64")
    cimage = np.copy(image)

    (imageH, imageW) = cimage.shape[:2]
    (kernelH, kernelW) = fkernel.shape[:2]

    pad = (kernelH - 1) // 2
    border = cv2.copyMakeBorder(fimage, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
    output = np.zeros((imageH, imageW), dtype="float64")

    for h in np.arange(pad, imageH + pad):
        for w in np.arange(pad, imageW, + pad):
            mat = border[h - pad:h + pad +1, w - pad:w + pad + 1]
            conv = (mat * kernel).sum()
            output[h - pad, w - pad] = conv

    return output

def main():

    loadDirectory = str(sys.argv[1])
    outputDirectory = str(sys.argv[2])
    numRows = int(sys.argv[3])
    numCols = int(sys.argv[4])
    alphaVal = float(sys.argv[5])
    betaVal = float(sys.argv[6])
    kernelVals = np.array([[sys.argv[7], sys.argv[8], sys.argv[9]], [sys.argv[10], sys.argv[11], sys.argv[12]], [sys.argv[13], sys.argv[14], sys.argv[15]]], dtype="float")

    argCount = len(sys.argv)
    if argCount < 7:
        print("ERROR: less than 7 parameters.")
        exit(1)
    if argCount < 7 + numRows * numCols:
        print("ERROR: less than 7 + rowCount and colCount parameters")
        exit(1)

    image = cv2.imread(loadDirectory, flags=cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("ERROR: Could not open the image.")
        exit(1)

    outputImage = applyFilter(image, kernelVals)
    outputArr = []
    output = cv2.convertScaleAbs(outputImage, np.array(outputArr), alphaVal, betaVal)

    cv2.imwrite(outputDirectory, output)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()