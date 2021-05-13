add_library(ellipsegnet_model OBJECT ellipsegnet_model.cpp)
target_include_directories(ellipsegnet_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ellipsegnet_model ${OpenCV_LIBS} ${TORCH_LIBRARIES})

add_library(ellipsegnet_data OBJECT dataset.cpp)
target_include_directories(ellipsegnet_data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ellipsegnet_data ${OpenCV_LIBS} ${TORCH_LIBRARIES} transformations)
