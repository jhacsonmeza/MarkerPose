cmake_minimum_required(VERSION 3.10)
project(MarkerPose)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_subdirectory(src/transformations)
add_subdirectory(src/utils)
add_subdirectory(src/ellipsegnet)
add_subdirectory(src/superpoint)


add_executable(robot_test robot_test.cpp src/MarkerPose/markerpose_model.cpp)
target_link_libraries(robot_test ${OpenCV_LIBS} ${TORCH_LIBRARIES} superpoint_model ellipsegnet_model utilities)

add_executable(train_superpoint train_superpoint.cpp)
target_link_libraries(train_superpoint ${OpenCV_LIBS} ${TORCH_LIBRARIES} superpoint_model superpoint_data utilities)

add_executable(train_ellipsegnet train_ellipsegnet.cpp)
target_link_libraries(train_ellipsegnet ${OpenCV_LIBS} ${TORCH_LIBRARIES} ellipsegnet_model ellipsegnet_data utilities)
