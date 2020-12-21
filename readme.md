## Iris Recognition System



1. Acquire Image: the iris dataset was provided in the AWS s3 cloud, to access the dataset we required an AWS educate account and AWS CLI to get multiple files.
2. Iris Segmentation: The main step in iris segmentation is to find the inner circular boundary and outer circular boundary. To do so we use the Hough transform algorithm for circle detection. In this the circle candidates are produced by voting in the Hough parameter space and then selecting local maxima in an accumulator matrix. In our code we use an inbuilt function from OpenCV to get the inner and outer circles.
Then, the inner and outer circle boundaries are used to ‘unwrap’ the iris region to a rectangle.
3. Analyze or encode Texture: A texture filter like, gabor filter is applied at fixed grids on the rectangle to generate the same size iris code for any image
4. Match Texture Encodings to verify or Recognize identity: To match the texture encoding we use hamming distance. If X and Y represents two binary patterns, the hamming distance is calculated by taking a XOR of the binary patterns and summing them up and the summation is divided by the total number of bits.


Following is the process to run the source code:

    #To simply run the network measure on the saved data
    python iris_recognition_lga4000.py
    python iris_recognition_lga2200.py	


##### Additional Information:
1. Dataset is private


##### Contributors:
Jasmeet Narang  
Shirish Mecheri Vogga
