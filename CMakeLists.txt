cmake_minimum_required(VERSION 3.0.2)
project(camera_lidar_cali)

add_compile_options(-std=c++14)

find_package(	catkin REQUIRED COMPONENTS
  		cv_bridge
  		pcl_conversions
  		pcl_ros
  		roscpp
  		rospy
  		sensor_msgs
  		std_msgs
  		image_transport
)

find_package(PCL REQUIRED)
find_package(OpenCV)



link_directories(${PCL_LIBRARY_DIRS}
		 ${OpenCV_INCLUDE_DIRS}
		 ${ROS_INCLUDE_DIRS}
		)

catkin_package(
 		CATKIN_DEPENDS 
 		roscpp 
 		rospy 
 		std_msgs
)
include_directories(
  		${catkin_INCLUDE_DIRS}
  		${PCL_INCLUDE_DIRS}
  		${OpenCV_INCLUDE_DIRS}
)

add_definitions(-fpermissive -g)

add_executable(camera_lidar_cali src/camera_lidar_cali.cpp)


target_link_libraries( camera_lidar_cali 
			${PCL_LIBRARIES} 
			${OpenCV_LIBS} 
			${catkin_LIBRARIES})


