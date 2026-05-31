#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/solver/state_converter.h>
#include <ocp_solver/solver/switched_model_reference_manager.h>

#include <string>
#include <vector>

namespace ocp_constraint_body_contact {

  class ContactPointSearchConstraint final : public ocs2::StateInputConstraint {
  public:
    ContactPointSearchConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                 size_t contactIndex,
                                 const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                                 const std::string& meshFile);
    ~ContactPointSearchConstraint() override = default;
    ContactPointSearchConstraint* clone() const override { return new ContactPointSearchConstraint(*this); }

    bool isActive(ocs2::scalar_t time) const override { return contactSearchEnabled_; }
    size_t getNumConstraints(ocs2::scalar_t time) const override;
    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation& preComp) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::vector_t& input,
                                                                   const ocs2::PreComputation& preComp) const override;

  private:
    ContactPointSearchConstraint(const ContactPointSearchConstraint& rhs);
    Eigen::Vector3d nearestNormalInParentFrame(const Eigen::Vector3d& pointInParentFrame) const;

    const ocp_solver::StateConverter<ocs2::scalar_t>* stateConverterPtr_;
    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_;
    const size_t contactIndex_;
    const bool contactSearchEnabled_;
    std::vector<Eigen::Vector3d> vertices_;
    std::vector<Eigen::Vector3d> normals_;
  };

}
