#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/solver/state_converter.h>
#include <ocp_solver/solver/switched_model_reference_manager.h>

#include <string>
#include <vector>

namespace ocp_constraint_body_contact {

  class AssumedSurfaceContactConstraint final : public ocs2::StateInputConstraint {
  public:
    struct Config {
      Config(){};
      Eigen::Vector3d contactNormalInContactFrame = Eigen::Vector3d::UnitZ();
      Eigen::Vector3d surfaceNormalInContactFrame = -Eigen::Vector3d::UnitZ();
      ocs2::scalar_t normalWeightScale = 50.0;
      ocs2::scalar_t covarianceRegularization = 1e-6;
      ocs2::scalar_t normalForceRegularization = 1e-6;
      ocs2::scalar_t ellipseSafetyMargin = 0.0;
    };
    AssumedSurfaceContactConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                    size_t contactIndex,
                                    const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                                    const std::string& meshFile,
                                    Config config=Config());

    ~AssumedSurfaceContactConstraint() override = default;
    AssumedSurfaceContactConstraint* clone() const override { return new AssumedSurfaceContactConstraint(*this); }

    bool isActive(ocs2::scalar_t time) const override;
    size_t getNumConstraints(ocs2::scalar_t time) const override { return n_constraints; }
    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation& preComp) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::vector_t& input,
                                                                   const ocs2::PreComputation& preComp) const override;
    const Eigen::Vector3d& getNormalInContactFrame() const { return normalInContactFrame_; }
    const Eigen::Vector3d& getGeometricCenterInContactPlane() const { return geometricCenterInContactPlane_; }
    const Eigen::Matrix3d& getEllipseMetricInContactFrame() const { return ellipseMetric_; }
    Eigen::Vector3d computePressureCenterInContactFrame(
        const Eigen::Matrix3d& contactRotation,
        const Eigen::Matrix<ocs2::scalar_t, 6, 1>& localWorldAlignedWrench) const;

  private:
    AssumedSurfaceContactConstraint(const AssumedSurfaceContactConstraint& rhs);
    void computeWeightedGeometry();

    const ocp_solver::StateConverter<ocs2::scalar_t>* stateConverterPtr_;
    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_;
    const size_t contactIndex_;
    static const int n_constraints = 1;
    Config config_;
    std::vector<Eigen::Vector3d> vertices_;
    Eigen::Vector3d normalInContactFrame_ = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d surfaceNormalInContactFrame_ = -Eigen::Vector3d::UnitZ();
    Eigen::Vector3d geometricCenter_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d geometricCenterInContactPlane_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d ellipseMetric_ = Eigen::Matrix3d::Identity();
  };

}
