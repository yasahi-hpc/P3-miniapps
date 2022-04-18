#ifndef __PARSER_HPP__
#define __PARSER_HPP__

#include <cstdio>
#include <string>
#include <string.h>

struct Parser {
  int num_threads_ = 1;
  int teams_ = 1;
  int device_ = 0;
  int ngpu_ = 1;
  char *file_ = "data.dat";

  Parser() = delete;
  Parser(int argc, char **argv) {
    for(int i = 0; i < argc; i++) {
      if((strcmp(argv[i], "-f") == 0) || (strcmp(argv[i], "--file") == 0)) {
        file_ = argv[++i];
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
                  
      if((strcmp(argv[i], "-dm") == 0) || (strcmp(argv[i], "--device_map") == 0)) {
        char *str;
        int local_rank;
                                
        if((str = getenv("SLURM_LOCALID")) != NULL) {
          local_rank = atoi(str);
          device_ = local_rank % ngpu_;
        }
                
        if((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
          local_rank = atoi(str);
          device_ = local_rank % ngpu_;
        }
                         
        if((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK")) != NULL) {
          local_rank = atoi(str);
          device_ = local_rank % ngpu_;
        }
                                 
        continue;
      }
    }
  }
  ~Parser() {}
};

#endif
