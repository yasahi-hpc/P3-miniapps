#ifndef __PARSER_HPP__
#define __PARSER_HPP__

#include <cstdio>
#include <string>
#include <string.h>
#include <vector>

struct Parser {
  std::vector<size_t> shape_;
  std::vector<int> topology_;
  int nbiter_ = 1000;

  Parser() = delete;
  Parser(int argc, char **argv) {
    shape_.resize(3);
    topology_.resize(3);

    size_t nx = 128, ny = 128, nz = 128;
    int px = 2, py = 2, pz = 2;

    for(int i = 0; i < argc; i++) {
      if((strcmp(argv[i], "-nx") == 0) || (strcmp(argv[i], "--nx") == 0)) {
        nx = static_cast<size_t>( atoi(argv[++i]) );
        continue;
      }

      if((strcmp(argv[i], "-ny") == 0) || (strcmp(argv[i], "--ny") == 0)) {
        ny = static_cast<size_t>( atoi(argv[++i]) );
        continue;
      }

      if((strcmp(argv[i], "-nz") == 0) || (strcmp(argv[i], "--nz") == 0)) {
        nz = static_cast<size_t>( atoi(argv[++i]) );
        continue;
      }
      if((strcmp(argv[i], "-px") == 0) || (strcmp(argv[i], "--px") == 0)) {
        px = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-py") == 0) || (strcmp(argv[i], "--py") == 0)) {
        py = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-pz") == 0) || (strcmp(argv[i], "--pz") == 0)) {
        pz = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-nbiter") == 0) || (strcmp(argv[i], "--nbiter") == 0)) {
        nbiter_ = atoi(argv[++i]);
        continue;
      }
    }
    shape_ = std::vector<size_t>({nx, ny, nz});
    topology_ = std::vector<int>({px, py, pz});
  }
  ~Parser() {}
};

#endif
