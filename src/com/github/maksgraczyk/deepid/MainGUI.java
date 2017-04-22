/*
    DeepID: research project about determining similarities between human faces by using a neural network
    Copyright (C) 2017 Maksymilian Graczyk.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

package com.github.maksgraczyk.deepid;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.*;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;

//Main window
public class MainGUI {
    private JPanel panel1; //panel containing all controls
    private JButton takePictureButton; //"Take a face sample"
    private JButton makeAFaceIdentificationAttemptButton; //"Make a face identification attempt"
    private JButton takePictureFromFileButton; //"Take a face sample from file"
    private JLabel statusLabel; //label indicating current status
    private JButton processAllRemainingImagesButton; //"Process all images"
    private JButton initializeTheModelButton; //"Initialize the model"

    public static MultiLayerNetwork model; //neural network model
    private static JFrame frame; //GUI frame

    public static ArrayList<String> labels = new ArrayList<>(); //labels (users' names) list

    private int imagesProcessed = 0; //number of images already sent to neural network
    private int imagesToProcess = 0; //number of images waiting to be processed

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME);} //OpenCV files have to be loaded when the app is run

    //Face detection (NOT recognition) function; it uses OpenCV
    public static Rect[] detectFaces(BufferedImage image)
    {
        CascadeClassifier detector = new CascadeClassifier(Thread.currentThread().getContextClassLoader().getResource("haarcascade_frontalface_alt.xml").getFile());
        Mat mat = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
        byte[] data = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        mat.put(0, 0, data);
        MatOfRect detections = new MatOfRect();
        detector.detectMultiScale(mat, detections);

        return detections.toArray();
    }

    //Called when "Take a face sample from a file" is clicked and the user is chosen
    public void processFile(String user)
    {
        JOptionPane.showMessageDialog(frame, "You'll be asked to select a picture file. Choose the one which is a picture containing ONE face ONLY, directed ahead to the camera.", "DeepID", JOptionPane.INFORMATION_MESSAGE);
        FileDialog dialog = new FileDialog(frame, "Choose a picture file", FileDialog.LOAD);
        dialog.setFile("*.jpg; *.jpeg; *.png");
        dialog.setVisible(true);
        File[] file = dialog.getFiles();
        if (file.length > 0)
        {
            try {
                BufferedImage image = ImageIO.read(file[0]);
                Rect[] array = detectFaces(image); //calls the face detection function
                if (array.length == 1) /*one face detected*/ {
                    BufferedImage bi2 = new BufferedImage(150, 150, BufferedImage.TYPE_INT_RGB); //picture of the detected face must have the size 150x150 pixels
                    Graphics gr = bi2.createGraphics();
                    gr.drawImage(image.getSubimage(array[0].x, array[0].y, array[0].width, array[0].height), 0, 0, 150, 150, null); //resize to 150x150
                    gr.dispose();
                    File[] files = new File("ToProcess/" + user + "/").listFiles();
                    ImageIO.write(bi2, "png", new File("ToProcess/" + user + "/" + (files == null ? 1 : files.length+1) + ".png")); //save the processed image to the file
                    updateImagesToProcess(1,1);
                }
                else if (array.length == 0) /*no faces detected*/JOptionPane.showMessageDialog(frame, "No faces have been detected. Are you sure the face is clearly visible?", "DeepID", JOptionPane.ERROR_MESSAGE);
                else /*more than one face detected*/ JOptionPane.showMessageDialog(frame, "There is more than 1 face in the picture. Please choose an image which contains ONE face ONLY.", "DeepID", JOptionPane.ERROR_MESSAGE);
            }
            catch (Exception ex)
            {
                JOptionPane.showMessageDialog(frame, "An error has occured when processing the selected file. Please try again.", "DeepID", JOptionPane.ERROR_MESSAGE);
                ex.printStackTrace();
            }
        }
    }

    //Called when "Take a face sample" is clicked and the user is chosen
    public void startPicture(String user)
    {
        JOptionPane.showMessageDialog(frame, "DeepID will start capturing pictures 5 seconds after clicking \"OK\". Please look closely to the camera then, so the computer detects your face\n(you'll be able to check it by looking at the preview).", "DeepID", JOptionPane.INFORMATION_MESSAGE);
        Picture picture = new Picture(this, user);
        picture.show(); //open the webcam preview
    }

    private MainGUI() {
        takePictureButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                NameChoice choice = new NameChoice(MainGUI.this, false, model == null);
                choice.show(); //open the user choice window
            }
        });
        makeAFaceIdentificationAttemptButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Identification identification = new Identification();
                identification.show(); //open the webcam preview for identification purposes
            }
        });
        takePictureFromFileButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                NameChoice choice = new NameChoice(MainGUI.this, true, model == null);
                choice.show(); //open the user choice window
            }
        });
        processAllRemainingImagesButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (imagesToProcess > 0) fit();
                else JOptionPane.showMessageDialog(frame, "There are no images to be processed! Please take some samples: press \"Take a face sample\" or/and \"Take a face sample from a file\".", "DeepID", JOptionPane.WARNING_MESSAGE);
            }
        });
        initializeTheModelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int answer = JOptionPane.showConfirmDialog(frame, "Are you sure you want to initialize the model? Once you do it, you won't be able to add any new users.", "DeepID", JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE);
                if (answer == JOptionPane.YES_OPTION) /*user clicked "Yes"*/ new Thread(initialiseTask).start(); //initializing model may be time-consuming, let it run in the background
            }
        });
    }

    //Files counting function
    private int countFiles(String path)
    {
        int result = 0;

        File file = new File(path);
        File[] files = file.listFiles();

        for (File f : files)
        {
            if (f.isDirectory()) result += countFiles(f.getAbsolutePath());
            else result += 1;
        }

        return result;
    }

    //Neural network model training function
    private void fit()
    {
        Runnable r = new Runnable() {
            @Override
            public void run() {
                statusLabel.setText("Status: Processing...");
                makeAFaceIdentificationAttemptButton.setEnabled(false);
                takePictureButton.setEnabled(false);
                takePictureFromFileButton.setEnabled(false);
                processAllRemainingImagesButton.setEnabled(false);
                try {
                    ImageRecordReader reader = new ImageRecordReader(150, 150, 3, new ParentPathLabelGenerator());
                    reader.initialize(new FileSplit(new File("ToProcess/"))); //read files from the "ToProcess" folder
                    int count = countFiles("ToProcess/");
                    DataSetIterator dataIter = new RecordReaderDataSetIterator(reader, count);
                    while (dataIter.hasNext()) { //iterate through each file
                        //Normalize the data got from the image file
                        DataNormalization normalization = new NormalizerMinMaxScaler();
                        DataSet set = dataIter.next();
                        normalization.fit(set);
                        normalization.transform(set);
                        //Send the data to the model
                        MainGUI.model.fit(set);
                        //Update appropriate variables and GUI labels
                        updateImagesProcessed(imagesToProcess);
                        updateImagesToProcess(-1, imagesToProcess);
                    }

                    reader.close();
                }
                catch (Exception e) {
                    statusLabel.setText("Status: An error has occured. However, a new attempt can be made.");
                    e.printStackTrace();
                }

                //Clear the "ToProcess" folder
                try {
                    File file = new File("ToProcess/");
                    for (File f : file.listFiles(File::isDirectory)) {
                        File[] array = f.listFiles();
                        for (int i = 0; i < array.length; i++) {
                            array[i].delete();
                        }
                    }
                }
                catch (Exception e)
                {
                    e.printStackTrace();
                }

                makeAFaceIdentificationAttemptButton.setEnabled(true);
            }
        };

        new Thread(r).start(); //run everything in the background
    }

    private Runnable initialiseTask = new Runnable() { //Model initialisation function
        @Override
        public void run() {
            File[] files = new File("ToProcess/").listFiles(File::isDirectory);

            if (files == null)
            {
                JOptionPane.showMessageDialog(frame, "There are no users! You will be able to create them when taking samples: press \"Take a face sample\" or \"Take a face sample from a file\".", "DeepID", JOptionPane.WARNING_MESSAGE);
                return;
            }

            statusLabel.setText("Status: Initializing the model...");
            makeAFaceIdentificationAttemptButton.setEnabled(false);
            takePictureButton.setEnabled(false);
            takePictureFromFileButton.setEnabled(false);
            processAllRemainingImagesButton.setEnabled(false);
            initializeTheModelButton.setEnabled(false);

            //Generate labels from the directory structure in ToProcess (created when taking samples)

            for (File f : files)
            {
                labels.add(f.getName());
            }

            Collections.sort(labels);

            int count = files.length;

            //Neural network model:
            //Learning rate: 0.02
            //Weight init: RELU (standard in most cases)
            //Optimization algorithm: STOCHASTIC_GRADIENT_DESCENT (standard in most cases)
            //Gradient normalization: Renormalize L2 per layer
            //Updater: SGD (standard in most cases)
            //Input data: RGB image 150x150 pixels
            //Layer 0: convolution layer, kernel size [5,5], stride [2,2], 100 filters applied
            //Layer 1: subsampling layer (MAX), kernel size [10,10], stride [10,10]
            //Layer 2: convolution layer, kenrel size [2,2], stride [1,1], 10 filters applied
            //Layer 3: subsampling layer (MAX), kernel size [5,5], stride [5,5]
            //Layer 4: convolution layer, kernel size [1,1], stride [1,1], 10 filters applied
            //Layer 5: dense layer, 1500 parameters
            //Layer 6: output layer, NEGATIVELOGLIKELIHOOD loss function (standard in most cases), SOFTMAX activation (giving scores easily convertible to percentage ones), number of outputs equal to number of users entered
            MultiLayerConfiguration conf =
                    new NeuralNetConfiguration.Builder()
                        .seed(123)
                        .iterations(1)
                        .regularization(true).l2(1e-3)
                        .learningRate(0.02)
                        .weightInit(WeightInit.RELU)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                        .updater(Updater.SGD)
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                .nIn(3)
                                .stride(2, 2)
                                .nOut(100)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(10, 10)
                                .stride(10, 10)
                                .build())
                        .layer(2, new ConvolutionLayer.Builder(2, 2)
                                .stride(1, 1)
                                .nOut(10)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                                .kernelSize(5, 5)
                                .stride(5, 5)
                                .build())
                        .layer(4, new ConvolutionLayer.Builder(1, 1)
                                .stride(1, 1)
                                .nOut(10)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(5, new DenseLayer.Builder()
                                .activation(Activation.RELU)
                                .nOut(1500)
                                .build())
                        .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX)
                                .nOut(count)
                                .build())
                            .setInputType(InputType.convolutional(150, 150, 3))
                            .backprop(true).pretrain(false)
                        .build();

            model = new MultiLayerNetwork(conf);
            model.init();

            try {
                UIServer server = UIServer.getInstance();
                StatsStorage storage = new InMemoryStatsStorage();
                server.attach(storage); //set up the monitoring webpage (at localhost:9000)
                model.setListeners(new StatsListener(storage), new IterationListener() {
                    int iterations = 0;
                    @Override
                    public boolean invoked() {
                        return false;
                    }

                    @Override
                    public void invoke() {

                    }

                    @Override
                    public void iterationDone(Model model, int i) {
                        iterations += 1;
                        //If there're less than 10 iterations done or the score of the model (lower the score, the more accurate results) is equal to or more than 0.2 and there're less than 1000 iterations, process the data through the model again
                            if (iterations < 10 || (model.score() >= 0.20 && iterations < 1000)) model.fit();
                    }
                });
            }
            catch (Exception e) {
                e.printStackTrace();
            }

            statusLabel.setText("Status: " + imagesProcessed + " image(s) has/have been processed so far. " + imagesToProcess + " image(s) ready to be processed.");
            takePictureButton.setEnabled(true);
            takePictureFromFileButton.setEnabled(true);
            makeAFaceIdentificationAttemptButton.setEnabled(true);
            processAllRemainingImagesButton.setEnabled(true);
            initializeTheModelButton.setEnabled(false);
        }
    };

    public void updateImagesToProcess(int multiplier, int number)
    {
        imagesToProcess += multiplier*number;
        if (model == null) statusLabel.setText("Status: The model is not ready. " + imagesToProcess + " image(s) ready to be processed.");
        else statusLabel.setText("Status: " + imagesProcessed + " image(s) has/have been processed so far. " + imagesToProcess + " image(s) ready to be processed.");
    }

    public void updateImagesProcessed(int number)
    {
        imagesProcessed += number;
        statusLabel.setText("Status: " + imagesProcessed + " image(s) has/have been processed so far. " + imagesToProcess + " image(s) ready to be processed.");
    }

    public static void main(String[] args) {
        frame = new JFrame("DeepID");
        frame.setContentPane(new MainGUI().panel1);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        //Clean and create a ToProcess folder
        //ToProcess folder contains directories with names indicating users in which image files are stored
        try {
            File file = new File("ToProcess/");
            boolean success = file.mkdirs();

            if (!success && !file.exists()) //If creating the directory fails
            {
                JOptionPane.showMessageDialog(frame, "The app couldn't create a ToProcess folder which is required to store all information about users along with samples. The program will now close.", "DeepID", JOptionPane.ERROR_MESSAGE);
                System.exit(0);
            }

            for (File f : file.listFiles())
            {
                for (File f2 : f.listFiles())
                {
                    f2.delete();
                }

                f.delete();
            }
            file.delete();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
}
