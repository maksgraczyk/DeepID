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
import org.opencv.core.*;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.Timer;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.TimerTask;

//Webcam preview (for taking samples purposes)
public class Picture {
    private JPanel panel; //panel containing all controls
    private JButton cancelButton; //"Cancel"
    private JLabel statusLabel; //label indicating the current status of the taking samples procedure
    private JLabel picturePanel; //panel containg the webcam preview

    private String user; //user chosen

    private JFrame frame; //GUI frame
    private Webcam webcam;

    private int samplesToTake = 5; //number of samples to take
    private int time; //seconds left before the taking samples procedure begins
    private boolean doing = false;

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); } //OpenCV files have to be loaded when the app is run

    //Updating the webcam preview function
    private TimerTask updatePreviewTask = new TimerTask() {
        @Override
        public void run() {
            BufferedImage bi = webcam.getImage();

            //Detect all faces in the preview and mark them
                Rect[] rect = MainGUI.detectFaces(bi);
                for (Rect rectangles : rect) {
                    Graphics gr = bi.createGraphics();
                    gr.drawRect(rectangles.x, rectangles.y, rectangles.width, rectangles.height);
                    gr.dispose();
                }

            BufferedImage bi2 = new BufferedImage(picturePanel.getWidth(), picturePanel.getHeight(), BufferedImage.TYPE_INT_RGB);
            Graphics gr = bi2.createGraphics();
            gr.drawImage(bi, 0, 0, picturePanel.getWidth(), picturePanel.getHeight(), null);
            gr.dispose();

            picturePanel.setIcon(new ImageIcon(bi2)); //show the preview to the user
        }
    };

    //Saving a sample taken to the appropriate directory in ToProcess function
    private TimerTask saveSampleTask = new TimerTask() {
        @Override
        public void run() {
            if (!doing) { //Don't run the function if it's already called
                doing = true;
                if (samplesToTake > 0) {
                    BufferedImage bi = webcam.getImage();

                    BufferedImage bi2 = new BufferedImage(150, 150, BufferedImage.TYPE_INT_RGB);

                    //Detect all face samples in the image taken
                    CascadeClassifier detector = new CascadeClassifier(Thread.currentThread().getContextClassLoader().getResource("haarcascade_frontalface_alt.xml").getFile());
                    Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
                    byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
                    mat.put(0, 0, data);
                    MatOfRect detections = new MatOfRect();
                    detector.detectMultiScale(mat, detections);

                    Rect[] array = detections.toArray();

                    if (array.length == 1) { //If there's only one face detected, resize the sample to 150x150 pixels and save it
                        Graphics gr = bi2.createGraphics();
                        gr.drawImage(bi.getSubimage(array[0].x, array[0].y, array[0].width, array[0].height), 0, 0, 150, 150, null);
                        gr.dispose();
                        try {
                            File[] files = new File("ToProcess/" + user + "/").listFiles();
                            ImageIO.write(bi2, "png", new File("ToProcess/" + user + "/" + (files == null ? 1 : files.length + 1) + ".png"));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                        parent.updateImagesToProcess(1, 1);
                        samplesToTake -= 1;
                    }
                }

                if (samplesToTake == 0) {
                    saveSampleTimer.cancel();
                    saveSampleTimer = null;
                    hide();
                }

                doing = false;
            }
            }
    };

    //User has 5 seconds to prepare the face to be taken. This function counts these 5 seconds.
    private TimerTask countdownTask = new TimerTask() {
        @Override
        public void run() {
            time -= 1;
            if (time == 0)
            {
                statusLabel.setText("Taking pictures...");
                samplesToTake = 4;
                countdownTimer.cancel();
                countdownTimer = null;
                saveSampleTimer.schedule(saveSampleTask, 1500, 1500);
            }
            else statusLabel.setText("Get ready...(" + time + " s)");
        }
    };

    private Timer updatePreviewTimer;
    private Timer saveSampleTimer;
    private Timer countdownTimer;

    private MainGUI parent; //reference to the main window

    public Picture(MainGUI parent, String user) {
        frame = new JFrame("Webcam preview");
        frame.setContentPane(panel);
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        frame.pack();
        frame.setLocationRelativeTo(null);
        this.user = user;
        this.parent = parent;
        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                hide();
            }
        });
    }

    public void show()
    {
        picturePanel.setText("");
        frame.setVisible(true);
        webcam = Webcam.getDefault();
        webcam.open(); //open the webcam
        updatePreviewTimer = new Timer();
        saveSampleTimer = new Timer();
        countdownTimer = new Timer();
        updatePreviewTimer.schedule(updatePreviewTask, 40, 40); //run the timer which updates the webcam preview
        time = 5;
        countdownTimer.schedule(countdownTask, 1000, 1000); //run the timer which counts 5 seconds before actual taking samples
    }

    public void hide()
    {
        try {
            updatePreviewTimer.cancel();
            updatePreviewTimer = null;
        }
        catch (Exception e) {

        }

        try
        {
            countdownTimer.cancel();
            countdownTimer = null;
        }
        catch (Exception e) {

        }

        try {
            saveSampleTimer.cancel();
            saveSampleTimer = null;
        }
        catch (Exception e) {

        }

        try {
            webcam.close();
            webcam = null;
        }
        catch (Exception e) {

        }

        frame.setVisible(false);
    }
}
