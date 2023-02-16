#ifndef field_h
#define field_h

#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <memory>
#include <numeric>
#include <algorithm>
#include <random>
#include <tuple>

template <typename T>
class Field {
    public:
        Field(size_t N, std::array<T, 3> h, T initVal=0.0) : N_m(N), h_m(h), f_m(std::vector<T>(N*N*N)){
            hInv_m[0] = 1.0 / h[0];
            hInv_m[1] = 1.0 / h[1];
            hInv_m[2] = 1.0 / h[2];

            init(initVal);
        }

        struct Index {
            typedef std::tuple<size_t,size_t,size_t> idx_tuple_t;
            Index(size_t i, size_t j, size_t k) : idx({i,j,k}), n(this->N_m) {}
            Index(size_t raveled_idx) : idx(unravel(raveled_idx)) {}

            inline T ravel(){return std::get<0>(idx)*n*n + std::get<1>(idx)*n + std::get<2>(idx);}
            inline idx_tuple_t unravel(size_t raveled_idx){
                size_t i,j,k;
                i = raveled_idx / (n*n);
                j = raveled_idx / n;
                k = raveled_idx % n;
                return {i,j,k};
            }

            idx_tuple_t idx;
            size_t n;
        };

        void init(T val){
			// Create a random number generator
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<T> dist(0, 20);
			// Fill the vector with random numbers
			std::generate(f_m.begin(), f_m.end(), [&](){ return dist(gen); });
        }

        std::array<T, 3> get_hInv() const {return hInv_m;};

        T operator()(size_t i, size_t j, size_t k) const {return f_m[index(i,j,k)];};

        T& operator()(size_t i, size_t j, size_t k) {return f_m[index(i,j,k)];};

        void print(size_t slice_k) {
            for(size_t i = 0; i < N_m; ++i){
                for(size_t j = 0; j < N_m; ++j){
                    std::cout << std::setprecision(2) << f_m[index(i,j,slice_k)] << " ";
                }
                std::cout << '\n';
            }
            std::cout << "\n\n" << std::endl;
        }

    

    private:
        inline size_t index(size_t i, size_t j, size_t k) const {
            return i*N_m*N_m + j*N_m + k;
        }

        // Number of grid points in each dimension
        const size_t N_m;
        // Mesh widths in each dimension
        const std::array<T, 3> h_m;
        std::array<T, 3> hInv_m;
        // Field data
        std::vector<T> f_m;
};

#endif // field_h
