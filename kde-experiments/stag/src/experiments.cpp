#include "experiments.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include "logging.h"
#include "data.h"
#include "KDE/kde.h"

class SimpleTimer {
public:
  SimpleTimer() = default;

  void start() {
    t1 = std::chrono::high_resolution_clock::now();
  };

  void stop() {
    t2 = std::chrono::high_resolution_clock::now();
  };

  StagInt elapsed_ms() {
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    return time_ms.count();
  }

  void report_time() {
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    LOG_INFO("Time: " << time_ms.count() << " milliseconds." << std::endl);
  };

private:
  std::chrono::time_point<std::chrono::system_clock> t1;
  std::chrono::time_point<std::chrono::system_clock> t2;
};

// Return build time (ms), query time (ms), relative error
std::vector<StagReal> single_trial(DenseMat* data, DenseMat* query, DenseMat* true_kdes, StagReal a,
                  StagReal min_mu, StagInt K1, StagReal K2_constant, StagInt offset) {
  // Initialise the timer variables
  auto timer = SimpleTimer();

  timer.start();
  stag::CKNSGaussianKDE ckns_data_structure = stag::CKNSGaussianKDE(
      data, a, min_mu, K1, K2_constant, offset);
  timer.stop();
  StagReal init_time_ms = timer.elapsed_ms();

  timer.start();
  std::vector<StagReal> estimates = ckns_data_structure.query(query);
  timer.stop();
  StagReal query_time_ms = timer.elapsed_ms();
  StagReal time_per_query = query_time_ms / query->rows();

  StagReal total_ratio = 0;
  StagReal total_error = 0;
  for (auto i = 0; i < query->rows(); i++)  {
    if (true_kdes->coeff(i, 0) < 0.000001) {
      continue;
    }
    StagReal this_error = (estimates[i] - true_kdes->coeff(i, 0)) / true_kdes->coeff(i, 0);
    total_ratio += abs(this_error);
    total_error += this_error;
  }
  StagReal rel_error = total_ratio / query->rows();

  return {init_time_ms, time_per_query, rel_error};
}


// Return build time (ms), query time (ms), relative error
std::vector<StagReal> stag_default(DenseMat* data, DenseMat* query, DenseMat* true_kdes, StagReal a) {
  // Initialise the timer variables
  auto timer = SimpleTimer();

  timer.start();
  stag::CKNSGaussianKDE ckns_data_structure = stag::CKNSGaussianKDE(data, a);
  timer.stop();
  StagReal init_time_ms = timer.elapsed_ms();

  timer.start();
  std::vector<StagReal> estimates = ckns_data_structure.query(query);
  timer.stop();
  StagReal query_time_ms = timer.elapsed_ms();
  StagReal time_per_query = query_time_ms / query->rows();

  StagReal total_ratio = 0;
  StagReal total_error = 0;
  for (auto i = 0; i < query->rows(); i++)  {
    if (true_kdes->coeff(i, 0) < 0.000001) {
      continue;
    }
    StagReal this_error = (estimates[i] - true_kdes->coeff(i, 0)) / true_kdes->coeff(i, 0);
    total_ratio += abs(this_error);
    total_error += this_error;
  }
  StagReal rel_error = total_ratio / query->rows();

  return {init_time_ms, time_per_query, rel_error};
}

// Function to strip trailing zeros
std::string stripTrailingZeros(const std::string& str) {
  size_t end = str.find_last_not_of('0');
  if (end != std::string::npos && str[end] == '.')
    end--; // If decimal point is the last character, keep it
  return str.substr(0, end + 1);
}

void run_experiment(const std::string& data_filename,
                    const std::string& results_filename,
                    StagReal mu) {
  // Initialise the timer variables
  auto timer = SimpleTimer();

  // Load the train dataset
  DenseMat data = load_hdf5(data_filename, "train");

  // Load the query dataset
  DenseMat query_data = load_hdf5(data_filename, "test");


  // Create ground truth data name
  std::string base_name = "kde.test.gaussian";
  std::ostringstream oss;
  oss << std::fixed << mu;
  std::string mu_str = stripTrailingZeros(oss.str());
  std::string attribute_name = base_name + mu_str.substr(1, std::string::npos);

  // Load the true estimates
  DenseMat true_kdes = load_hdf5(data_filename, attribute_name);

  StagReal bandwidth = get_bandwidth(data_filename, attribute_name);
  StagReal a = 0.5 / SQR(bandwidth);

  LOG_INFO("Dataset size: " << data.rows() << " x " << data.cols() << std::endl);
  LOG_INFO("Query dataset size: " << query_data.rows() << " x " << query_data.cols() << std::endl);
  LOG_INFO("True KDEs size: " << true_kdes.rows() << " x " << true_kdes.cols() << std::endl);
  LOG_INFO("bandwidth: " << bandwidth << ", a: " << a << std::endl);

  std::ofstream result_file;
  result_file.open(results_filename);

  // Work out how many total trials there will be.
  StagReal relative_error_cutoff = 0.2;
  StagReal running_time_cutoff = 1;
  StagReal decrease_factor = 0.7;
  StagInt n = data.rows();

  std::vector<StagReal> mus;
  auto max_mu = (StagReal) 1;
  StagReal next_mu = MAX(1 / (StagReal) n, 0.01 * mu);
  while (next_mu <= max_mu) {
    mus.push_back(next_mu);
    next_mu /= decrease_factor;
    next_mu /= decrease_factor;
  }
  if (mus.empty() || mus.back() != max_mu) mus.push_back(max_mu);
  assert(!mus.empty());

  StagInt max_offset = 15;
  StagInt min_offset = -15;

  StagInt max_k1 = 5 * (StagInt) log((StagReal) n);
  std::vector<StagInt> k1s;
  StagInt this_k1 = max_k1;
  while (this_k1 >= 1) {
    k1s.push_back(this_k1);

    if (this_k1 > 1 / decrease_factor) {
      this_k1 = (StagInt) (this_k1 * decrease_factor);
    } else {
      if (this_k1 == 1) {
        this_k1 = 0;
      } else {
        this_k1 = 1;
      }
    }
  }
  if (k1s.back() != 1) k1s.push_back(1);

  StagReal max_k2_constant = 5 * log((StagReal) n);
  StagReal this_k2_constant = max_k2_constant;
  std::vector<StagReal> k2_constants;
  while (this_k2_constant >= 0.05) {
    k2_constants.push_back(this_k2_constant);

    // Change the k2 constant twice as fast as everything else
    this_k2_constant *= decrease_factor;
    this_k2_constant *= decrease_factor;
  }

  StagInt total_trials = (StagInt) mus.size() * (max_offset - min_offset) * (StagInt) k1s.size() * (StagInt) k2_constants.size();

  StagInt trial_number = 1;
  result_file << "mu, offset, k1, k2_constant, rel_err, query_time, build_time" << std::endl;

  for (StagReal this_mu : mus) {
    // Iterate through increasing offsets until the approximation factor is not good enough.
    for (StagInt offset = 0; offset < max_offset; offset += 1) {
      bool last_k1_good = true;
      for (StagInt K1 : k1s) {
        bool this_k1_still_good = last_k1_good;
        last_k1_good = false;
        for (StagReal K2_constant : k2_constants) {
          if (trial_number == 1 || trial_number % 100 == 0) {
            LOG_INFO("Beginning trial " << trial_number << " / " << total_trials << std::endl);
          }
          if ((StagInt) floor(log2((StagReal) n * this_mu)) + offset >= 0) {
            if (this_k1_still_good) {
              result_file << this_mu << ", " << offset << ", " << K1 << ", " << K2_constant << ", ";
              result_file.flush();
              std::vector<StagReal> stats = single_trial(&data, &query_data, &true_kdes,
                                                         a, this_mu, K1, (StagReal) K2_constant, offset);
              result_file << stats[2] << ", " << stats[1] << ", " << stats[0];
              result_file << std::endl;
              result_file.flush();

              if (stats[2] >= relative_error_cutoff) {
                this_k1_still_good = false;
              } else {
                last_k1_good = true;
              }
            }
          }
          trial_number++;
        }
      }
    }

    // Iterate through decreasing offsets until the running time is too slow.
    for (StagInt offset = -1; offset > min_offset; offset -= 1) {
      bool last_k1_good = true;
      for (auto k1_iter = k1s.rbegin(); k1_iter != k1s.rend(); ++k1_iter) {
        StagInt K1 = *k1_iter;
        bool this_k1_still_good = last_k1_good;
        last_k1_good = false;
        for (auto k2_iter = k2_constants.rbegin(); k2_iter != k2_constants.rend(); ++k2_iter) {
          StagReal K2_constant = *k2_iter;
          if (trial_number == 1 || trial_number % 100 == 0) {
            LOG_INFO("Beginning trial " << trial_number << " / " << total_trials << std::endl);
          }
          if ((StagInt) floor(log2((StagReal) n * this_mu)) + offset >= 0) {
            if (this_k1_still_good) {
              result_file << this_mu << ", " << offset << ", " << K1 << ", " << K2_constant << ", ";
              result_file.flush();
              std::vector<StagReal> stats = single_trial(&data, &query_data, &true_kdes,
                                                         a, this_mu, K1, (StagReal) K2_constant, offset);
              result_file << stats[2] << ", " << stats[1] << ", " << stats[0];
              result_file << std::endl;
              result_file.flush();

              if (stats[1] >= running_time_cutoff) {
                this_k1_still_good = false;
              } else {
                last_k1_good = true;
              }
            }
          }
          trial_number++;
        }
      }
    }
  }

  result_file.close();
}
