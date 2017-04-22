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

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

//User choice window
public class NameChoice {
    private JComboBox pleaseChooseTheUserComboBox; //combo box with available users
    private JButton OKButton; //"OK"
    private JButton cancelButton; //"Cancel"
    private JPanel choicePanel; //panel containing all controls
    private boolean cancelled = false; //indicates if the user cancelled taking a face sample

    private JFrame frame; //GUI frame

    //If the model is not initialized, a user can be added by entering its name to the combo box. If it is initialized, no new users can be added, but the users entered so far can still be chosen.

    public NameChoice(MainGUI parent, boolean fromFile, boolean newUsersAllowed) {
        frame = new JFrame("User choice");
        frame.setContentPane(choicePanel);
        frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
        frame.pack();
        frame.setLocationRelativeTo(null);
        pleaseChooseTheUserComboBox.setEditable(newUsersAllowed);
        cancelButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                cancelled = true;
                frame.setVisible(false);
            }
        });
        OKButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (getUserName() == null || getUserName().equals("")) JOptionPane.showMessageDialog(frame, "The user's name field must not be empty!", "DeepID", JOptionPane.WARNING_MESSAGE);
                else {
                    frame.setVisible(false);
                    new File("ToProcess/" + getUserName() + "/").mkdirs(); //create a directory indicating the new user in ToProcess
                    if (!fromFile) parent.startPicture(getUserName());
                    else parent.processFile(getUserName());
                }
            }
        });

        //Get the users list by checking directories in ToProcess
        try {
            File file = new File("ToProcess/");
            File[] files = file.listFiles(File::isDirectory);
            if (files != null)
            {
                for (File f : files)
                {
                    pleaseChooseTheUserComboBox.addItem(f.getName());
                }
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    public void show()
    {
        frame.setVisible(true);
    }

    //Getting the currently selected user name function
    public String getUserName()
    {
        if (!cancelled)
        {
            if (pleaseChooseTheUserComboBox.getSelectedItem() != null) return pleaseChooseTheUserComboBox.getSelectedItem().toString();
            else return null;
        }
        else return null;
    }
}
