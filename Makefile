# Compiler to use
CXX = gcc

BUILD_DIR := build
SRC_DIR := src

# Compiler flags
# -Wall: Enable all standard warnings
# -Wextra: Enable extra warnings
# -std=c++17: Use C++17 standard (adjust as needed)
# -I/path/to/eigen-3.4.0: Specify the path to the Eigen root directory
#                         (e.g., where the 'Eigen' folder is located)
CXXFLAGS = -Wall -Wextra -std=c++17 -I /usr/include/eigen3/ # Adjust this path

# Source files
SRCS = src/test_cholesky_decomp.cc
OBJS = build/test_cholesky_decomp.o

# Object files (automatically generated from source files)
# OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = my_eigen_program

$(BUILD_DIR)/%.o: $(BUILD_DIR)/%.cc | $(BUILD_DIR)
	$(CC) $(CXXFLAGS) -c $< -o $@


# Default target: build the executable
all: $(TARGET)

# Rule to link object files into the executable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

# Rule to compile C++ source files into object files
build/test_cholesky_decomp.o: src/test_cholesky_decomp.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target: remove generated files
# clean:
# 	rm -f $(OBJS) $(TARGET)