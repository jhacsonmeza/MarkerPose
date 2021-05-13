add_library(superpoint_model OBJECT superpoint_model.cpp)
target_include_directories(superpoint_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(superpoint_model ${OpenCV_LIBS} ${TORCH_LIBRARIES})

add_library(superpoint_data OBJECT dataset.cpp)
target_include_directories(superpoint_data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(superpoint_data ${OpenCV_LIBS} ${TORCH_LIBRARIES} transformations)
