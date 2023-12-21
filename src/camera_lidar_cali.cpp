#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>
#include <sstream>
#include <stdio.h>
#include <string>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>



class CameraLidarCal
{
  public:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher color_cloud_pub_=nh_.advertise<sensor_msgs::PointCloud2>("livox/colord_cloud",1);
    ros::Subscriber cloud_sub_=nh_.subscribe("livox/lidar",1,&CameraLidarCal::cloudCb,this);

    const int save_flag=1;
    int color_intensity_threshold_ = 5;
    const int density = 5;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;

    const cv::Mat input_image;
  //所需变量
    
    
    CameraLidarCal()
      : it_(nh_)
      {
        image_sub_=it_.subscribe("/usb_cam/image_raw",1, &CameraLidarCal::imageCb, this);
      }

      ~CameraLidarCal()
      {
        
      }

      void imageCb(const sensor_msgs::ImageConstPtr& msg)
      {
      cv_bridge::CvImagePtr cv_ptr;
      try
        {
          cv_ptr=cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);

        }
        catch (cv_bridge::Exception& e)
        {
          ROS_ERROR("cv_bridge exception: %s", e.what());
          return;    
        }
        input_image=cv_ptr->image;
        ROS_INFO("Image converted!!");
      }

      void cloudCb(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) 
      {  
        ROS_INFO("Color cloud ing!!");
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw_lidar_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PCLPointCloud2 cloud_msg_temp;

        const cv::Mat rgb_img;
            // 相机内参
        int width_, height_;
        cv::Mat camera_matrix_ = (cv::Mat_<double>(3,3)<< 669.9496 ,   0.       , 283.96895,
                                                      0.          ,   672.21534, 219.00958,
                                                      0.          ,   0.       , 1.       );
        cv::Mat dist_coeffs_ = (cv::Mat_<double>(1,4)<< -0.444346, 0.195529, 0.004070, 0.004716, 0.000000);
        cv::Mat init_extrinsic_;

        float fx_ = camera_matrix_.at<double>(0, 0);
        float cx_ = camera_matrix_.at<double>(0, 2);
        float fy_ = camera_matrix_.at<double>(1, 1);
        float cy_ = camera_matrix_.at<double>(1, 2);
        float k1_ = dist_coeffs_.at<double>(0, 0);
        float k2_ = dist_coeffs_.at<double>(0, 1);
        float p1_ = dist_coeffs_.at<double>(0, 2);
        float p2_ = dist_coeffs_.at<double>(0, 3);
        float k3_ = dist_coeffs_.at<double>(0, 4);

        pcl_conversions::toPCL(*cloud_msg, cloud_msg_temp);
        pcl::fromPCLPointCloud2(cloud_msg_temp, *raw_lidar_cloud);
        if (input_image.type() == CV_8UC3) {
          rgb_img=input_image;
        } 
        else if (input_image.type() == CV_8UC1) {
          cv::cvtColor(input_image, rgb_img, cv::COLOR_GRAY2BGR);
        } //确保输入类型正确

        std::vector<cv::Point3f> pts_3d;
        for (size_t i = 0; i < raw_lidar_cloud->size(); i += density) {//定义的密度
          pcl::PointXYZI point = raw_lidar_cloud->points[i];
          float depth = sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));//计算深度
          if (depth > 0.1 && depth < 50 &&
              point.intensity >= color_intensity_threshold_) { //如果深度在2到50之间并且强度有效
            pts_3d.emplace_back(cv::Point3f(point.x, point.y, point.z));//将当前点放入容器中
          }
        }
        Eigen::AngleAxisd rotation_vector3;//定义3*1旋转向量
        Vector6d extrinsic_params;
        extrinsic_params[0]=1.5261;
        extrinsic_params[1]=-1.48429;
        extrinsic_params[2]=0.00109644;
        extrinsic_params[3]=0.0687349;
        extrinsic_params[4]=-0.243097;
        extrinsic_params[5]=-0.0241099;
        rotation_vector3 =
            Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
        cv::Mat camera_matrix =
            (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
        cv::Mat distortion_coeff =
            (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
        cv::Mat r_vec =
            (cv::Mat_<double>(3, 1)
                << rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
            rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
            rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
        cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
                        extrinsic_params[4], extrinsic_params[5]);
        std::vector<cv::Point2f> pts_2d;
        cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,
                          pts_2d);
        int image_rows = rgb_img.rows;-
        int image_cols = rgb_img.cols;
        for (size_t i = 0; i < pts_2d.size(); i++) {
          if (pts_2d[i].x >= 0 && pts_2d[i].x < image_cols && pts_2d[i].y >= 0 &&
              pts_2d[i].y < image_rows) {
            cv::Scalar color =
                rgb_img.at<cv::Vec3b>((int)pts_2d[i].y, (int)pts_2d[i].x);
            if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
              continue;
            }
            if (pts_3d[i].x > 100) {
              continue;
            }
            pcl::PointXYZRGB p;
            p.x = pts_3d[i].x;
            p.y = pts_3d[i].y;
            p.z = pts_3d[i].z;
            // p.a = 255;
            p.b = color[0];
            p.g = color[1];
            p.r = color[2];
            color_cloud->points.push_back(p);
          }
        }
        color_cloud->width = color_cloud->points.size();
        color_cloud->height = 1;
        sensor_msgs::PointCloud2 color_cloud_msg;
        pcl::toROSMsg(*color_cloud, color_cloud_msg);
        color_cloud_msg.header.frame_id = "livox_frame";
        color_cloud_msg.header.stamp=ros::Time::now();
        ROS_INFO("Color cloud success, publish !!");
        color_cloud_pub_.publish(color_cloud_msg);        
      }
};
 

int main(int argc, char** argv)
{
  ros::init(argc,argv,"color_the_cloud");
  ROS_INFO("started!!");
  CameraLidarCal clc;
  ros::spin();
}
