cmake_minimum_required(VERSION 3.16)
project(GomokuQt6 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/mainwindow.h
    src/mainwindow.cpp
    src/boardwidget.h
    src/boardwidget.cpp
    src/gamemodel.h
    src/gamemodel.cpp
    src/gamecontroller.h
    src/gamecontroller.cpp
    src/aiclient.h
    src/aiclient.cpp
    src/python_ai_process.h
    src/python_ai_process.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Widgets)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/ai_stdin.py
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/ai_stdin.py
)