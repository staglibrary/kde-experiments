//
// This file is provided as part of the STAG library and released under the MIT
// license.
//
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "graph.h"
#include "utility.h"
#include "graphio.h"

/**
 * Parse a single content line of an edgelist file. This method assumes that
 * the line is not a comment, and tries to parse either by splitting on commas
 * or whitespace.
 *
 * @return a triple representing the edge (u, v, weight).
 * @throw std::invalid_argument the line cannot be parsed
 */
EdgeTriplet parse_edgelist_content_line(std::string line) {
  // List the possible delimiters for the elements on the line
  std::vector<std::string> delimiters{",", " ", "\t"};

  // Split the line to extract the edge information
  int num_tokens_found = 0;
  int u;
  int v;
  double weight;

  // Try splitting by each delimiter in turn
  size_t split_pos = 0;
  std::string token;
  for (const std::string &delimiter: delimiters) {
    // If we have not found any delimiters of the previous types yet,
    // then try this one.
    if (num_tokens_found == 0) {
      while ((split_pos = line.find(delimiter)) != std::string::npos) {
        // Extract the portion of the line up to the delimiter
        token = line.substr(0, split_pos);
        line.erase(0, split_pos + delimiter.length());

        // If the token has length 0, then skip
        if (token.length() == 0) continue;

        // Parse the token as the appropriate data type - int or double
        // throws an exception if the token cannot be parsed.
        if (num_tokens_found == 0) u = std::stoi(token);
        if (num_tokens_found == 1) v = std::stoi(token);
        if (num_tokens_found == 2) weight = std::stod(token);

        // Increase the counter of the number of tokens found
        num_tokens_found++;
      }
    }
  }

  // Extract the final token in the line
  if (num_tokens_found == 1) {
    v = std::stoi(line);
    num_tokens_found++;
  } else if (num_tokens_found == 2) {
    try {
      // Try extracting the weight from the rest of the line, but ignore any
      // errors - the weight might not be there.
      weight = std::stod(line);
      num_tokens_found++;
    } catch (std::exception &e) {
      // Ignore this
    }
  }

  // Check that we have exactly two or three elements in the split line
  if (num_tokens_found < 2 || num_tokens_found > 3) {
    throw std::invalid_argument("Wrong number of tokens on edgelist line.");
  }

  // If we have exactly two elements on the line, then add the weight 1.
  if (num_tokens_found == 2) {
    weight = 1;
  }

  // Return the triple
  return {u, v, weight};
}

stag::Graph stag::load_edgelist(std::string &filename) {
  // Attempt to open the provided file
  std::ifstream is(filename);

  // If the file could not be opened, throw an exception
  if (!is.is_open()) {
    throw std::runtime_error(std::strerror(errno));
  }

  // We will construct a vector of triples in order to construct the final
  // adjacency matrix
  std::vector<EdgeTriplet> non_zero_entries;

  // Read the file in one line at a time
  stag_int number_of_vertices = 0;
  std::string line;
  EdgeTriplet this_edge;
  while (stag::safeGetline(is, line)) {
    if (line[0] != '#' && line[0] != '/' && line.length() > 0) {
      try {
        // This line of the input file isn't a comment, parse it.
        this_edge = parse_edgelist_content_line(line);

        // Add two edges to the adjacency matrix in order to keep it symmetric.
        non_zero_entries.emplace_back(this_edge);
        non_zero_entries.emplace_back(
            EdgeTriplet(this_edge.col(), this_edge.row(), this_edge.value()));

        // Update the number of vertices to be the maximum of the column and row
        // indices.
        number_of_vertices = std::max(number_of_vertices, this_edge.col() + 1);
        number_of_vertices = std::max(number_of_vertices, this_edge.row() + 1);
      } catch (std::invalid_argument &e) {
        // Re-throw any parsing errors
        throw(std::runtime_error(e.what()));
      }
    }
  }

  // Close the input file stream
  is.close();

  // Update the adjacency matrix from the triples constructed from the input file.
  SprsMat adj_mat(number_of_vertices, number_of_vertices);
  adj_mat.setFromTriplets(non_zero_entries.begin(), non_zero_entries.end());

  // Construct and return the graph object
  return stag::Graph(adj_mat);
}

void stag::save_edgelist(stag::Graph &graph, std::string &filename) {
  // Attempt to open the specified file
  std::ofstream os(filename);

  // If the file could not be opened, throw an exception
  if (!os.is_open()) {
    throw std::runtime_error(std::strerror(errno));
  }

  // Write header information to the output file.
  os << "# This file was automatically generated by the STAG library." << std::endl;
  os << "#   number of vertices = " << graph.number_of_vertices() << std::endl;
  os << "#   number of edges = " << graph.number_of_edges() << std::endl;

  // Iterate through the entries in the graph adjacency matrix, and write
  // the edgelist file
  const SprsMat* adj_mat = graph.adjacency();
  for (int k = 0; k < adj_mat->outerSize(); ++k) {
    for (SprsMat::InnerIterator it(*adj_mat, k); it; ++it) {
      // We only consider the 'upper triangle' of the matrix
      if (it.col() > it.row()) {
        os << it.row() << " " << it.col() << " " << it.value() << std::endl;
      }
    }
  }

  // Close the output file stream
  os.close();
}
