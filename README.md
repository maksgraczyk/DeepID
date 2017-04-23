# DeepID
Research project about determining similarities between human faces by using a neural network. It is a student project.

I have used IntelliJ IDEA Community Edition (which is under the Apache license: http://www.apache.org/licenses/LICENSE-2.0.html) as an IDE to create this project.

## Libraries used
* deeplearning4j (Maven: org.deeplearning4j:deeplearning4j-core **and** org.deeplearning4j:deeplearning4j-ui_2.10: used to create a neural network model
* nd4j (Maven: org.nd4j:nd4j-native **or** org.nd4j:nd4j-cuda-\*; see CUDA): a dependency of deeplearning4j
* webcam-capture written by Bartosz Firyn and contributors (Maven: com.github.sarxos:webcam-capture): used to access a computer webcam
* OpenCV: used to detect (**NOT** recognise) human faces in pictures, along with the haarcascade_frontalface_alt.xml resource

### Versions recommended
* deeplearning4j: 0.7.2
* nd4j: 0.7.2
* webcam-capture: 0.3.11
* OpenCV: 3.2.0

### OpenCV
If you want to use this source code, you'll need to build and put OpenCV to the project as a library. Version 3.2.0 is recommended. One of the tutorials how to do it is located at https://medium.com/@aadimator/how-to-set-up-opencv-in-intellij-idea-6eb103c1d45c (accessed on 22 April 2017).

## Requirements
* A computer with a webcam
* Java
* Libraries mentioned in "Libraries used"

## CUDA
By default, a CPU does all necessary computation related to the project. However, it is possible to use a CUDA-supported graphic card instead. This may increase the speed dramatically and is highly recommended.

To enable this option, change one of the Maven dependencies: nd4j-native to nd4j-cuda-\*. \* must be replaced by an appropriate version of CUDA you have installed in your device (e.g. 8.0).

## Neural network details
All details are included inside the source code.

## License
**The project itself is licensed under GNU GPL v3.**

License of libraries and other resources used:
* deeplearning4j is licensed under the Apache license.
* nd4j is licensed under the Apache license.
* webcam-capture is licensed under the MIT license.
* OpenCV is licensed under the 3-clause BSD license.
* haarcascade_frontalface_alt.xml file is licensed under the Intel License Agreement For Open Source Computer Vision Library.

All the licenses are available at appropriate LICENSE files and in the haarcascade_frontalface_alt.xml file.
