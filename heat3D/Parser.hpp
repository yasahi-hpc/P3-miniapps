#ifndef __PARSER_HPP__
#define __PARSER_HPP__

#include <cstdio>
#include <string>
#include <string.h>

struct Parser {
  int nx_ = 128;
  int ny_ = 128;
  int nz_ = 128;
  int nbiter_ = 1000;

  Parser() = delete;
  Parser(int argc, char **argv) {
    for(int i = 0; i < argc; i++) {
      if((strcmp(argv[i], "-nx") == 0) || (strcmp(argv[i], "--nx") == 0)) {
        nx_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-ny") == 0) || (strcmp(argv[i], "--ny") == 0)) {
        ny_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-nz") == 0) || (strcmp(argv[i], "--nz") == 0)) {
        nz_ = atoi(argv[++i]);
        continue;
      }

      if((strcmp(argv[i], "-nbiter") == 0) || (strcmp(argv[i], "--nbiter") == 0)) {
        nbiter_ = atoi(argv[++i]);
        continue;
      }
    }
  }
  ~Parser() {}
};

#endif
