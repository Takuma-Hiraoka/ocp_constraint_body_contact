#include "ocp_constraint_body_contact/contact_point_search_constraint.h"

#include <assimp_eigen/assimp_eigen.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace ocp_constraint_body_contact {

  ContactPointSearchConstraint::ContactPointSearchConstraint(
      const ocp_solver::SwitchedModelReferenceManager& referenceManager,
      size_t contactIndex,
      const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
      const std::string& meshFile)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      stateConverterPtr_(&stateConverter),
      referenceManagerPtr_(&referenceManager),
      contactIndex_(contactIndex),
      contactSearchEnabled_(stateConverter.getContactCandidate(contactIndex).searchContactPoint) {
    if (!contactSearchEnabled_) {
      return;
    }
    const assimp_eigen::MeshData mesh = assimp_eigen::loadMesh(meshFile);
    vertices_ = mesh.vertices;
    normals_ = mesh.normals;
    if (vertices_.empty()) {
      throw std::runtime_error("ContactPointSearchConstraint: mesh has no vertices: " + meshFile);
    }
    if (normals_.size() < vertices_.size()) {
      normals_.resize(vertices_.size(), Eigen::Vector3d::UnitZ());
    }
    for (Eigen::Vector3d& normal : normals_) {
      if (!normal.allFinite() || normal.squaredNorm() < 1e-12) {
        normal = Eigen::Vector3d::UnitZ();
      } else {
        normal.normalize();
      }
    }
  }

  ContactPointSearchConstraint::ContactPointSearchConstraint(const ContactPointSearchConstraint& rhs)
    : StateInputConstraint(rhs),
      stateConverterPtr_(rhs.stateConverterPtr_->clone()),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      contactIndex_(rhs.contactIndex_),
      contactSearchEnabled_(rhs.contactSearchEnabled_),
      vertices_(rhs.vertices_),
      normals_(rhs.normals_) {}

  size_t ContactPointSearchConstraint::getNumConstraints(ocs2::scalar_t time) const {
    if (!contactSearchEnabled_) {
      return 0;
    }
    return referenceManagerPtr_->isInContact(time, stateConverterPtr_->getContactCandidateIds()[contactIndex_]) ? 3 : 1;
  }

  Eigen::Vector3d ContactPointSearchConstraint::nearestNormalInParentFrame(const Eigen::Vector3d& pointInParentFrame) const {
    size_t nearestIndex = 0;
    double nearestDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < vertices_.size(); ++i) {
      const double distance = (vertices_[i] - pointInParentFrame).squaredNorm();
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = i;
      }
    }
    return normals_[nearestIndex];
  }

  ocs2::vector_t ContactPointSearchConstraint::getValue(ocs2::scalar_t time,
                                                        const ocs2::vector_t& state,
                                                        const ocs2::vector_t& input,
                                                        const ocs2::PreComputation&) const {
    const Eigen::Vector3d localVelocity = stateConverterPtr_->getContactPointLocalVelocity(input, contactIndex_);
    if (referenceManagerPtr_->isInContact(time, stateConverterPtr_->getContactCandidateIds()[contactIndex_])) {
      return localVelocity;
    }
    ocs2::vector_t value(1);
    const Eigen::Vector3d normal =
      nearestNormalInParentFrame(stateConverterPtr_->getContactPointLocalPosition(state, contactIndex_));
    value[0] = normal.dot(localVelocity);
    return value;
  }

  ocs2::VectorFunctionLinearApproximation ContactPointSearchConstraint::getLinearApproximation(
      ocs2::scalar_t time,
      const ocs2::vector_t& state,
      const ocs2::vector_t& input,
      const ocs2::PreComputation& preComp) const {
    ocs2::VectorFunctionLinearApproximation approx;
    approx.f = getValue(time, state, input, preComp);
    approx.dfdx = ocs2::matrix_t::Zero(approx.f.rows(), stateConverterPtr_->getStateVariableDim());
    approx.dfdu = ocs2::matrix_t::Zero(approx.f.rows(), stateConverterPtr_->getInputDim());
    const size_t inputIndex = stateConverterPtr_->getContactPointLocalVelocityStartIndex(contactIndex_);
    if (approx.f.rows() == 3) {
      approx.dfdu.block<3, 3>(0, inputIndex).setIdentity();
    } else {
      const Eigen::Vector3d normal =
        nearestNormalInParentFrame(stateConverterPtr_->getContactPointLocalPosition(state, contactIndex_));
      approx.dfdu.block<1, 3>(0, inputIndex) = normal.transpose();
    }
    return approx;
  }

}
