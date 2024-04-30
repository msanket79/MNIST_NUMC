cmake_minimum_required(VERSION 3.29.0)

project(MNIST_C LANGUAGES CXX )

find_package(OpenCV REQUIRED)

find_package(OpenMP REQUIRED)




set(SOURCES 
    main.cpp
    src/neuralNet.cpp
    modules/layers/src/affineLayer.cpp
    modules/layers/src/dropoutLayer.cpp
    modules/layers/src/layer.cpp
    modules/layers/src/reluLayer.cpp
    modules/lossFunctions/src/softmax.cpp
    modules/optimizers/src/adam.cpp
    modules/MNIST/src/readMNIST.cpp
    modules/MNIST/src/showMNIST.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")

include_directories(include ${OpenCV_INCLUDE_DIRS})

include_directories(include
                    modules/numC/include                    
                    modules/layers/include
                    modules/lossFunctions/include                    
                    modules/optimizers/include                    
                    modules/MNIST/include
                    modules/numC/openblas/include
                
)

add_executable(main ${SOURCES})

set_target_properties( main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin
)

target_link_libraries(main "E:/MNIST_NUMC/modules/numC/openblas/lib/libopenblas.lib" ${OpenCV_LIBS} OpenMP::OpenMP_CXX)