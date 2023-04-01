#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <string_view>
#include <iostream>

#include "ros/ros.h"

#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int32MultiArray.h"

// kreis = 0
// dreieck = 1
// stern = 2
// regenschirm = 3
// quadrat = 4
// nopass = -1
std::vector<std::string> forms = {"kreis", "dreieck", "stern", "regenschirm", "quadrat"};
std::vector<std::string> formsRes = {};
std::vector<int> formsResStation = {};
std::vector<int> formsResMenu = {};
std::vector<std::string> colors = {"red", "green", "blue"};
std::vector<std::string> colorsRes = {};
std::vector<int> colorsResStation = {};
std::vector<int> colorsResMenu = {};


void menuCallback(const std_msgs::Int32MultiArrayConstPtr &array)
{
    ROS_INFO("Menu");
    ROS_INFO_STREAM(std::string{"new Message"});
    int i = 0;
    // print all the remaining numbers
    for (std::vector<int>::const_iterator it = array->data.begin(); it != array->data.end(); ++it)
    {
        if (i < 5)
        {
            if (*it == -1)
            {
                // ROS_INFO("Gelb");
            }
            else
            {
                ROS_INFO_STREAM(std::string(forms[*it]));
                formsResMenu.insert(formsResMenu.end(),*it);
            }
        }
        else{
            if (*it == -1)
            {
                // ROS_INFO("No Match");
            }
            else
            {
                ROS_INFO_STREAM(std::string(colors[*it]));
                colorsResMenu.insert(colorsResMenu.end(),*it);
                
            }
        }
        i++;
    }
}
void stationCallback(const std_msgs::Int32MultiArrayConstPtr &array)
{   ROS_INFO("Station");
    ROS_INFO_STREAM(std::string{"new Message"});
    int i = 0;
    // print all the remaining numbers
    for (std::vector<int>::const_iterator it = array->data.begin(); it != array->data.end(); ++it)
    {
        if (i < 5)
        {
            if (*it == -1)
            {
                // ROS_INFO("Gelb");
            }
            else
            {
                ROS_INFO_STREAM(std::string(forms[*it]));
                formsResStation.insert(formsResStation.end(),*it);
            }
        }
        else{
            if (*it == -1)
            {
                // ROS_INFO("No Match");
            }
            else
            {
                ROS_INFO_STREAM(std::string(colors[*it]));
                colorsResStation.insert(colorsResStation.end(),*it);

            }

        }
        i++;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "objErkennungCPP");
    ros::NodeHandle n;
    ros::Subscriber subMenu = n.subscribe("ResMenu", 1000, menuCallback);
    ros::Subscriber subStation = n.subscribe("ResStation", 1000, stationCallback);
    ros::spin();
}