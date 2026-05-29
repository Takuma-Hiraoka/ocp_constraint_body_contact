cmake_minimum_required(VERSION 3.16)
project(ocp_constraint_body_contact_sample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)

set(dependencies
  ament_index_cpp
  assimp_eigen
  mujoco_viewer
  ocp_constraint
  ocp_constraint_body_contact
  ocp_solver
  trajectory_logger
)

foreach(dep ${dependencies})
  find_package(${dep} REQUIRED)
endforeach()

add_executable(visualize src/visualize.cpp)

ament_target_dependencies(visualize
  ${dependencies}
  )

install(TARGETS
  visualize
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(reach_pose src/reach_pose.cpp)

ament_target_dependencies(reach_pose
  ${dependencies}
  )

install(TARGETS
  reach_pose
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY unitree_ros
  DESTINATION share/${PROJECT_NAME}
  )

ament_package()
