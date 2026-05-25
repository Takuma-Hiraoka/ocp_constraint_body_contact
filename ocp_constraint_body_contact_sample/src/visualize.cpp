#include <mujoco_viewer/mujoco_viewer.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>

int main() {
  mujoco_viewer::Viewer viewer;
  viewer.viewModel(ament_index_cpp::get_package_share_directory("ocp_constraint_body_contact_sample") + "/unitree_ros/robots/g1_description/g1_29dof.xml");
  while (viewer.isOpen())
    {
      viewer.updateScene();
      viewer.drawScene();
      viewer.pollEvents();
    }

  return 0;
}
