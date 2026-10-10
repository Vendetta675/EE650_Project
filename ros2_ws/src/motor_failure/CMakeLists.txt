cmake_minimum_required(VERSION 3.8)
project(motor_failure)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED)
find_package(gz-transport13 REQUIRED)

# 1. Compile the library outside of the testing block
# 2. Name the target exactly what Gazebo expects
add_library(motor_failure SHARED src/MotorFailureSystem.cpp)
target_link_libraries(motor_failure
  gz-sim8::gz-sim8
  gz-plugin2::gz-plugin2
  gz-transport13::gz-transport13
)

# 3. Install directly to the primary 'lib' directory
install(TARGETS motor_failure
  DESTINATION lib)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
