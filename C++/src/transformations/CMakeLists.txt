file(GLOB SOURCES *.cpp)
add_library(transformations STATIC ${SOURCES})

target_include_directories(transformations PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(transformations ${OpenCV_LIBS})