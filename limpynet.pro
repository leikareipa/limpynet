TARGET = limpynet
TEMPLATE = app
CONFIG -= app_bundle
CONFIG -= qt
CONFIG += console c++11

OBJECTS_DIR = generated_files
MOC_DIR = generated_files
UI_DIR = generated_files

SOURCES += src/main.cpp \
    src/nnetwork/nnetwork.cpp \
    src/cmd_line/cmd_line.cpp \
    src/file/file.cpp \
    src/train_on/mnist/train_on_mnist.cpp \
    src/train_on/mnist/mnist_data.cpp

HEADERS  += src/nnetwork/nnetwork.h \
    src/common.h \
    src/types.h \
    src/cmd_line/cmd_line.h \
    src/file/file.h \
    src/train_on/train_on.h \
    src/train_on/mnist/mnist_data.h

# C++. For GCC/Clang/MinGW.
QMAKE_CXXFLAGS += -g
QMAKE_CXXFLAGS += -O2
QMAKE_CXXFLAGS += -Wall
QMAKE_CXXFLAGS += -ansi
QMAKE_CXXFLAGS += -pipe
QMAKE_CXXFLAGS += -pedantic
QMAKE_CXXFLAGS += -std=c++11
