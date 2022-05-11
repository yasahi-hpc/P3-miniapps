#ifndef __PARSER_HPP__
#define __PARSER_HPP__

#include <cstdio>
#include <string>
#include <string.h>
#include <vector>

struct Parser {
  std::vector<size_t> shape_;
  int nbiter_ = 1000;
  int num_threads_ = 1;
  int teams_ = 1;
  int device_ = 0;
  int ngpu_ = 1;

  Parser() = delete;
  Parser(int argc, char **argv) {
    shape_.resize(3);

    size_t nx = 128, ny = 128, nz = 128;

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

      if((strcmp(argv[i], "-nbiter") == 0) || (strcmp(argv[i], "--nbiter") == 0)) {
        nbiter_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-t") == 0) || (strcmp(argv[i], "--num_threads") == 0)) {
        num_threads_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "--teams") == 0)) {
        teams_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-d") == 0) || (strcmp(argv[i], "--device") == 0)) {
        device_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-ng") == 0) || (strcmp(argv[i], "--num_gpus") == 0)) {
        ngpu_ = atoi(argv[++i]);
        continue;
      }
    }
    shape_ = std::vector<size_t>({nx, ny, nz});
  }
  ~Parser() {}
};

#endif
