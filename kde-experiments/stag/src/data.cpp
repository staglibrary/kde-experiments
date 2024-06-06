#include "data.h"
#include "logging.h"

StagReal get_bandwidth(const std::string& filename, const std::string& attribute_name) {
  H5Easy::File file(filename, HighFive::File::ReadWrite);
  H5Easy::Attribute attr = file.getAttribute(attribute_name);
  auto data = attr.read<std::vector<StagReal>>();
  assert(data.size() > 1);
  return data.at(1);
}

DenseMat load_hdf5(const std::string& filename, const std::string& dataset) {
  H5Easy::File file(filename, HighFive::File::ReadWrite);
  return H5Easy::load<DenseMat>(file, dataset);
}

std::vector<std::string> list_datasets_hdf5(const std::string& filename) {
  H5Easy::File file(filename, HighFive::File::ReadWrite);
  return file.listObjectNames();
}
