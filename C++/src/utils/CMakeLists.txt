file(GLOB SOURCES *.cpp)
add_library(utilities STATIC ${SOURCES})

target_include_directories(utilities PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(utilities ${OpenCV_LIBS} ${TORCH_LIBRARIES})