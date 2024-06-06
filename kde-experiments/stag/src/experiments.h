#ifndef KDE_EXPERIMENTS_EXPERIMENTS_H
#define KDE_EXPERIMENTS_EXPERIMENTS_H

#include <string>
#include "definitions.h"

/**
 * Run a single experiment with the CKNS algorithm from STAG for
 * the given data file.
 *
 * @param data_filename
 * @param mu
 */
void run_experiment(const std::string& data_filename,
                    const std::string& results_filename,
                    StagReal mu);

void test_stag_default(const std::string& data_filename,
                       StagReal mu);

#endif //KDE_EXPERIMENTS_EXPERIMENTS_H
