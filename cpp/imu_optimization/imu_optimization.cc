//
// Created by yanhang on 2/6/17.
//

#include "imu_optimization.h"

namespace ridi {

SparseGrid::SparseGrid(const double *time_stamp, const int N,
                       const int variable_count,
                       const std::vector<int> *variable_ind) :
    kTotalCount(N), kVariableCount(variable_count) {
  alpha_.resize((size_t) kTotalCount);
  inverse_ind_.resize((size_t) kTotalCount);
  variable_ind_.resize((size_t) kVariableCount);
  if (variable_ind != nullptr) {
    CHECK_EQ(variable_ind->size(), kVariableCount);
    for (auto i = 0; i < kVariableCount; ++i) {
      variable_ind_[i] = (*variable_ind)[i];
    }
  } else {
    const int interval = kTotalCount / kVariableCount;
    for (int i = 0; i < kVariableCount; ++i) {
      variable_ind_[i] = (i + 1) * interval - 1;
    }
  }

  // Compute the interpolation weights and inverse indexing
  // y[i] = (1.0 - alpha[i]) * x[i-1] + alpha[i] * x[i]
  for (int j = 0; j <= variable_ind_[0]; ++j) {
    CHECK_GT(time_stamp[variable_ind_[0]] - time_stamp[0], std::numeric_limits<double>::epsilon()) << variable_ind_[0];
    alpha_[j] = (time_stamp[j] - time_stamp[0]) / (time_stamp[variable_ind_[0]] - time_stamp[0]);
  }

  for (int i = 1; i < variable_ind_.size(); ++i) {
    CHECK_GT(time_stamp[variable_ind_[i]] - time_stamp[variable_ind_[i - 1]], std::numeric_limits<double>::epsilon())
      << variable_ind_[i] << ' ' << variable_ind_[i - 1] << ' ' << time_stamp[variable_ind_[i]] << ' '
      << time_stamp[variable_ind_[i - 1]];
    for (int j = variable_ind_[i - 1] + 1; j <= variable_ind_[i]; ++j) {
      inverse_ind_[j] = i;
      alpha_[j] = (time_stamp[j] - time_stamp[variable_ind_[i - 1]]) /
          (time_stamp[variable_ind_[i]] - time_stamp[variable_ind_[i - 1]]);
    }
  }
}
} // namespace ridi
