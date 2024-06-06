#include <iostream>
#include "logging.h"
#include "experiments.h"


int main(int argc, char *argv[]) {
  if (argc < 4) {
    LOG_ERROR("Please provide the dataset and results filename as arguments, as well as the target mu." << std::endl);
    return -1;
  }

  run_experiment(argv[1], argv[2], std::stod(argv[3]));

  return 0;
}
