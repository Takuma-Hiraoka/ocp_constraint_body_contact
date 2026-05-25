#include <mujoco_viewer/mujoco_viewer.h>
#include <assimp_eigen/assimp_eigen.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  Eigen::Vector3d transformPoint(const mjtNum* xpos, const mjtNum* xmat, const Eigen::Vector3d& point) {
    return Eigen::Vector3d(
      xpos[0] + xmat[0] * point.x() + xmat[1] * point.y() + xmat[2] * point.z(),
      xpos[1] + xmat[3] * point.x() + xmat[4] * point.y() + xmat[5] * point.z(),
      xpos[2] + xmat[6] * point.x() + xmat[7] * point.y() + xmat[8] * point.z());
  }

  Eigen::Vector3d transformDirection(const mjtNum* xmat, const Eigen::Vector3d& direction) {
    return Eigen::Vector3d(
      xmat[0] * direction.x() + xmat[1] * direction.y() + xmat[2] * direction.z(),
      xmat[3] * direction.x() + xmat[4] * direction.y() + xmat[5] * direction.z(),
      xmat[6] * direction.x() + xmat[7] * direction.y() + xmat[8] * direction.z());
  }

  struct MeshGeomIds {
    int meshId = -1;
    int geomId = -1;
  };

  MeshGeomIds findMeshGeom(const mjModel* model, const std::string& meshName) {
    const int meshId = mj_name2id(model, mjOBJ_MESH, meshName.c_str());
    if (meshId < 0) {
      throw std::runtime_error("mesh not found in MuJoCo model: " + meshName);
    }
    for (int geomId = 0; geomId < model->ngeom; ++geomId) {
      if (model->geom_type[geomId] == mjGEOM_MESH && model->geom_dataid[geomId] == meshId) {
        return {meshId, geomId};
      }
    }
    throw std::runtime_error("geom using mesh not found: " + meshName);
  }

  Eigen::Vector3d transformAssimpPointToMujocoMeshFrame(const mjModel* model, int meshId, const Eigen::Vector3d& point) {
    const Eigen::Vector3d scale(model->mesh_scale[3 * meshId + 0],
                                model->mesh_scale[3 * meshId + 1],
                                model->mesh_scale[3 * meshId + 2]);
    const Eigen::Vector3d meshPos(model->mesh_pos[3 * meshId + 0],
                                  model->mesh_pos[3 * meshId + 1],
                                  model->mesh_pos[3 * meshId + 2]);
    const Eigen::Quaterniond meshQuat(model->mesh_quat[4 * meshId + 0],
                                      model->mesh_quat[4 * meshId + 1],
                                      model->mesh_quat[4 * meshId + 2],
                                      model->mesh_quat[4 * meshId + 3]);
    return meshQuat.conjugate() * (point.cwiseProduct(scale) - meshPos);
  }

  Eigen::Vector3d transformAssimpNormalToMujocoMeshFrame(const mjModel* model, int meshId, const Eigen::Vector3d& normal) {
    const Eigen::Quaterniond meshQuat(model->mesh_quat[4 * meshId + 0],
                                      model->mesh_quat[4 * meshId + 1],
                                      model->mesh_quat[4 * meshId + 2],
                                      model->mesh_quat[4 * meshId + 3]);
    return (meshQuat.conjugate() * normal).normalized();
  }

  void appendNormalLines(mujoco_viewer::Viewer& viewer,
                         const assimp_eigen::MeshData& mesh,
                         MeshGeomIds ids,
                         double length,
                         size_t maxLines) {
    const mjModel* model = viewer.model();
    const mjData* data = viewer.data();
    const mjtNum* xpos = data->geom_xpos + 3 * ids.geomId;
    const mjtNum* xmat = data->geom_xmat + 9 * ids.geomId;
    const size_t vertexCount = std::min(mesh.vertices.size(), mesh.normals.size());
    const size_t stride = std::max<size_t>(1, vertexCount / std::max<size_t>(1, maxLines));
    const float rgba[4] = {1.0f, 0.1f, 0.1f, 1.0f};

    for (size_t i = 0; i < vertexCount; i += stride) {
      if (!mesh.normals[i].allFinite() || mesh.normals[i].squaredNorm() == 0.0) continue;
      const Eigen::Vector3d vertex = transformAssimpPointToMujocoMeshFrame(model, ids.meshId, mesh.vertices[i]);
      const Eigen::Vector3d normal = transformAssimpNormalToMujocoMeshFrame(model, ids.meshId, mesh.normals[i]);

      const Eigen::Vector3d from = transformPoint(xpos, xmat, vertex);
      const Eigen::Vector3d to = from + length * transformDirection(xmat, normal).normalized();
      const mjtNum fromArray[3] = {from.x(), from.y(), from.z()};
      const mjtNum toArray[3] = {to.x(), to.y(), to.z()};

      mjvGeom* line = viewer.appendGeom();
      if (!line) break;
      mjv_initGeom(line, mjGEOM_LINE, nullptr, nullptr, nullptr, rgba);
      mjv_connector(line, mjGEOM_LINE, 2.0, fromArray, toArray);
    }
  }
}

int main() {
  const std::string packageShare = ament_index_cpp::get_package_share_directory("ocp_constraint_body_contact_sample");
  const std::string modelPath = packageShare + "/unitree_ros/robots/g1_description/g1_29dof.xml";
  const std::string meshName = "right_elbow_link";
  const std::string meshPath = packageShare + "/unitree_ros/robots/g1_description/meshes/" + meshName + ".STL";

  mujoco_viewer::Viewer viewer;
  viewer.viewModel(modelPath);

  const assimp_eigen::MeshData mesh = assimp_eigen::loadMesh(meshPath);
  const MeshGeomIds meshGeomIds = findMeshGeom(viewer.model(), meshName);

  while (viewer.isOpen())
    {
      viewer.updateScene();
      appendNormalLines(viewer, mesh, meshGeomIds, 0.03, 1200);
      viewer.drawScene();
      viewer.pollEvents();
    }

  return 0;
}
