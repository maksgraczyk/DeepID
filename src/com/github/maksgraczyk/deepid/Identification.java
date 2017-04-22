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

import com.github.sarxos.webcam.Webcam;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
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
import java.text.DecimalFormat;
import java.util.*;
import java.util.Timer;

//Webcam preview (for identification purposes)
public class Identification {
    private JPanel identificationPanel; //panel containing all controls
    private JButton cancelButton; //"Cancel"
    private JLabel picture; //webcam preview
    private JTextArea probabilityArea; //text area where classification results are printed
    private JFrame frame; //GUI frame
    private Webcam webcam;

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); } //OpenCV files have to be loaded when the app is run

    //Updating the webcam preview function
    private TimerTask refreshTask = new TimerTask() {
        @Override
        public void run() {
            BufferedImage bi = webcam.getImage();

            //Detect all faces in the preview and mark them
            CascadeClassifier detector = new CascadeClassifier(Thread.currentThread().getContextClassLoader().getResource("haarcascade_frontalface_alt.xml").getFile());
                Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
                byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
                mat.put(0, 0, data);
                MatOfRect detections = new MatOfRect();
                detector.detectMultiScale(mat, detections);

                for (Rect rectangles : detections.toArray()) {
                    Graphics gr = bi.createGraphics();
                    gr.drawRect(rectangles.x, rectangles.y, rectangles.width, rectangles.height);
                    gr.dispose();
                }

            BufferedImage bi2 = new BufferedImage(picture.getWidth(), picture.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics gr = bi2.createGraphics();
            gr.drawImage(bi, 0, 0, picture.getWidth(), picture.getHeight(), null);
            gr.dispose();

            picture.setIcon(new ImageIcon(bi2)); //show the preview to the user
        }
    };

    //Taking a face sample function
    private TimerTask takeSampleTask = new TimerTask() {
        @Override
        public void run() {
            BufferedImage bi = webcam.getImage();

            //Detect all face samples in the preview
                CascadeClassifier detector = new CascadeClassifier(Thread.currentThread().getContextClassLoader().getResource("haarcascade_frontalface_alt.xml").getFile());
                Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
                byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
                mat.put(0, 0, data);
                MatOfRect detections = new MatOfRect();
                detector.detectMultiScale(mat, detections);

                Rect[] array = detections.toArray();

                if (array.length == 1) { //If there's only one face detected, resize the sample to 150x150 pixels and send it to the model
                    BufferedImage bi2 = new BufferedImage(150, 150, BufferedImage.TYPE_INT_RGB);
                    Graphics gr = bi2.createGraphics();
                    gr.drawImage(bi.getSubimage(array[0].x, array[0].y, array[0].width, array[0].height), 0, 0, 150, 150, null);
                    gr.dispose();

                    try {
                        check(bi2); //sends the image to the model
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
        }
    };

    private Timer refreshTimer;
    private Timer takeSampleTimer;

    public Identification() {
        frame = new JFrame("Identification");
        frame.setContentPane(identificationPanel);
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        frame.pack();
        frame.setLocationRelativeTo(null);
        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hide();
            }
        });
    }

    //Sending the image to the model and getting the results function
    private void check(BufferedImage image) throws Exception
    {
        ImageIO.write(image, "png", new File("tmp.png")); //saves the image to the tmp.png file
        ImageRecordReader reader = new ImageRecordReader(150, 150, 3);
        reader.initialize(new FileSplit(new File("tmp.png"))); //reads the tmp.png file
        DataSetIterator dataIter = new RecordReaderDataSetIterator(reader, 1);
        while (dataIter.hasNext())
        {
            //Normalize the data from the file
            DataNormalization normalization = new NormalizerMinMaxScaler();
            DataSet set = dataIter.next();
            normalization.fit(set);
            normalization.transform(set);

            INDArray array = MainGUI.model.output(set.getFeatures(), false); //send the data to the model and get the results

            //Process the results and print them in an understandable format (percentage scores)
            String txt = "";

            DecimalFormat df = new DecimalFormat("#.00");

            for (int i = 0; i < array.length(); i++)
            {
                txt += MainGUI.labels.get(i) + ": " + (array.getDouble(i)*100 < 1 ? "0" : "") + df.format((array.getDouble(i)*100)) + "%\n";
            }

            probabilityArea.setText(txt);
        }

        reader.close();
    }

    private void prepare()
    {
        webcam = Webcam.getDefault();
        webcam.open(); //opens the webcam
        refreshTimer = new Timer();
        refreshTimer.schedule(refreshTask, 40, 40); //runs the webcam preview update timer
        takeSampleTimer = new Timer();
        takeSampleTimer.schedule(takeSampleTask, 1000, 1000); //runs the taking a face sample timer
    }

    private void dispose()
    {
        refreshTimer.cancel();
        refreshTimer = null;
        takeSampleTimer.cancel();
        takeSampleTimer = null;
        webcam.close();
        webcam = null;
    }

    public void show()
    {
        frame.setVisible(true);
        prepare();
    }

    public void hide()
    {
        frame.setVisible(false);
        dispose();
    }
}
