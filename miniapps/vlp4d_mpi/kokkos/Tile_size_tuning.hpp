#ifndef __TILE_SIZE_TUNING_HPP__
#define __TILE_SIZE_TUNING_HPP__

#include <map>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include <iostream>
#include <fstream>
#include <Kokkos_Core.hpp>
#include <sstream>
#include <iomanip>
#include <chrono>
 
struct Metric {
  double second_;
  std::vector<int> config_;
};

class TileSizeTuning {
  private:
    int max_tile_size_;
    const int DIM = 4;
        
    // Store the main results
    std::map<std::string, std::vector< Metric > > results_;
   
    // Problem size for the given kernel 
    std::map<std::string, std::vector<int> > size_configs_;
    
    // Possible tile sizes for the given kernel
    std::map<std::string, std::vector< std::vector<int> > > tile_configs_;

    // Best tile size for the given kernel
    std::map<std::string, std::vector<int> > best_tile_configs_;

    // Best seconds for the given kernel
    std::map<std::string, double> best_seconds_;
    
  public:
    TileSizeTuning() : max_tile_size_(0) {
      #if defined ( KOKKOS_ENABLE_CUDA )
        max_tile_size_ = 256;
      #endif
    }
    
    ~TileSizeTuning() {}

    /* Register the kernel metric to optimize */
    void registerKernel(const std::string name, const std::vector<int> &size_config) {
      assert(size_config.size() <= 4);
      std::vector<int> size_config_tmp(DIM, 1);

      for(size_t i=0; i<size_config.size(); i++) {
        size_config_tmp.at(i) = size_config.at(i);
      }
      size_configs_[name] = size_config;
     
      std::vector< std::vector<int> > all_tiles(DIM);
      for(int i=0; i<DIM; i++) {
        const int max_size = size_config[i];
        const int max_power = static_cast<int> ( std::log2(max_size+1) );
        std::vector<int> tile;
        int tile_size = 1;
        tile.push_back(tile_size);
        for(int power=0; power < max_power; power++) {
          tile_size *= 2;
          //if(tile_size <= max_size) { avoid the same size
          if(tile_size < max_size) {
            tile.push_back(tile_size);
          }
        }
        all_tiles[i] = tile;
      }
       
      std::vector< std::vector<int> > tile_config;
      for(auto it0: all_tiles[0]) {
        for(auto it1: all_tiles[1]) {
          for(auto it2: all_tiles[2]) {
            for(auto it3: all_tiles[3]) {
              std::vector<int> tile = {it0, it1, it2, it3};
              if(max_tile_size_ == 0) {
                tile_config.push_back(tile);
              } else {
                if(it0*it1*it2*it3 <= max_tile_size_) {
                  tile_config.push_back(tile);
                }
              }
            }
          }
        }
      }
      
      tile_configs_[name] = tile_config;
    }

    /* Pop the tile size to scan */
    bool pop(const std::string name, std::vector<int> &tile_config) {
      if(tile_configs_[name].empty()) {
        return false;
      } else {
        tile_config = tile_configs_[name].back();
        tile_configs_[name].pop_back();
        return true;
      }
    }

    /* Push the elapsed second with the given tile size */
    void push(const std::string name, const std::vector<int> &tile_config, const double second) {
      std::vector<int> config(DIM*2);
      std::vector<int> size_config = size_configs_[name];
     
      for(int i=0; i<DIM; i++) {
        config.at(i) = size_config.at(i);
        config.at(i+4) = tile_config.at(i);
      }
      
      Metric metric;
      metric.second_ = second;
      metric.config_ = config;
      
      results_[name].push_back( metric );
    }

    /* Save all the data to a single file */
    void toCsv(const std::string name, const int rank) {
      std::string filename = "opt_data/" + name + "_rank" + zfill(rank, 4) + ".csv";
      std::ofstream file(filename);
      if(!file) {
        std::cerr << "Failed to open the file" << std::endl;
        std::exit(1);
      }
      file << "Kernel,Nx,Ny,Nvx,Nvy,Tx,Ty,Tvx,Tvy,seconds" << "\n";
      for(auto result: results_) {
        std::string kernel_name = result.first;
        
        std::vector< Metric > metrics;
        metrics = result.second;
        for(auto metric: metrics) {
          file << kernel_name << ",";
          for(auto it: metric.config_) {
            file << it << ",";
          }
          file << metric.second_ << "\n";
        }
      }
    }

    void findBest(const std::string name) {
      std::vector< Metric > metrics = results_[name];
      std::vector< double > seconds;
      std::vector< std::vector<int> > tile_configs;
      for(auto metric: metrics) {
        seconds.push_back(metric.second_);
        std::vector<int> tile_config(std::begin(metric.config_)+4, std::end(metric.config_));
        tile_configs.push_back(tile_config);
      }

      double second_min = *std::min_element(seconds.begin(), seconds.end());
      auto   it = std::min_element(seconds.begin(), seconds.end());
      size_t min_index = std::distance(seconds.begin(), it);
      best_tile_configs_[name] = tile_configs.at(min_index);
      best_seconds_[name] = second_min;
    }

    template <class FunctorType>
    void scan(const std::string &name, const FunctorType &functor, const int rank) {
      auto start_total = std::chrono::high_resolution_clock::now();
      bool to_continue = true;
      std::chrono::high_resolution_clock::time_point start, end;
      int count = 0;
      while(to_continue) {
        std::vector<int> tile;
        to_continue = pop(name, tile);
        if(!tile.empty()) {
          start = std::chrono::high_resolution_clock::now(); 
          functor(tile);
          Kokkos::fence();
          end   = std::chrono::high_resolution_clock::now(); 

          double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
          push(name, tile, seconds);
          count++;
        }
      }
      findBest(name);
      printBest(name, rank);
      auto end_total = std::chrono::high_resolution_clock::now();
      double total_second = std::chrono::duration_cast<std::chrono::duration<double> >(end_total - start_total).count();

      std::stringstream ss;
      ss << "Auto tuning of " + name + " @ rank" << rank << " takes " << std::scientific << std::setprecision(15) << total_second << " [s] with " << count << " scans";
      std::cout << ss.str() << std::endl;
    }

    /* Return the best tile size for a given kernel based on the measured timing 
     * [Remark] Call this after using findBest()
     * */
    std::vector<int> bestTileSize(const std::string &name) {
      return best_tile_configs_[name];
    }

    double bestSeconds(const std::string &name) {
      return best_seconds_[name];
    }

    void printBest(const std::string &name, const int rank) {
      std::stringstream ss;
      std::vector<int> tile = bestTileSize(name);
      double second = bestSeconds(name);
      ss << "Best performance of " + name + " @ rank" << rank << " is " << std::scientific << std::setprecision(15) << second << " [s] with tiles " << 
            tile[0] << ", " << tile[1] << ", " << tile[2] << ", " << tile[3];
      std::cout << ss.str() << std::endl;
    }

    static std::string zfill(int n, int length = 3) {
      std::ostringstream s;
      s << std::setfill('0') << std::setw(length) << n;
      return s.str();
    }


};
#endif
