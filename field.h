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

#include <Eigen/Dense>

enum Dim {X, Y, Z};

template<typename T>
T relative_error(T x, T x_appr){ return std::abs((x - x_appr) / x);}

struct Index {
    typedef std::tuple<size_t,size_t,size_t> idx_tuple_t;
    Index(idx_tuple_t idx_arg, size_t N) : idx(idx_arg), n(N) {}
    Index(size_t raveled_idx) : idx(unravel(raveled_idx)) {}

    inline size_t ravel(){return std::get<0>(idx)*n*n + std::get<1>(idx)*n + std::get<2>(idx);}
    inline idx_tuple_t unravel(size_t raveled_idx){
        size_t i,j,k;
        i = raveled_idx / (n*n);
        j = raveled_idx / n;
        k = raveled_idx % n;
        return {i,j,k};
    }

    template<Dim D>
    inline Index get_shifted(size_t shift) {
        Index shifted_idx(idx, n);
        if constexpr (D == Dim::X) {
            std::get<0>(shifted_idx.idx) += shift;
        } else if constexpr (D == Dim::Y) {
            std::get<1>(shifted_idx.idx) += shift;
        } else if constexpr (D == Dim::Z) {
            std::get<2>(shifted_idx.idx) += shift;
        }
        return shifted_idx;
    }

    idx_tuple_t idx;
    size_t n;
};


template <typename T>
class Field {
    public:
        typedef Eigen::Vector3d vector3d_t;
        typedef Eigen::Matrix3d matrix3d_t;
        Field(size_t N, vector3d_t h) : N_m(N), h_m(h), f_m(std::vector<T>(N*N*N)){
            hInv_m[0] = 1.0 / h[0];
            hInv_m[1] = 1.0 / h[1];
            hInv_m[2] = 1.0 / h[2];
        }

        void init(T val){
			// Create a random number generator
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<T> dist(0, 20);
			// Fill the vector with random numbers
			std::generate(f_m.begin(), f_m.end(), [&](){ return dist(gen); });
        }

        T x2y2z2(T x, T y, T z) {
            return x*x*z+y*y+z*z*x;
        }

        T x2y2z2_deriv(T x, T y, T z) {
            return 2.0*x+2.0*z;
        }

        void init_x2y2z2(){
            for(size_t i = 0; i < N_m; ++i){
                for(size_t j = 0; j < N_m; ++j){
                    for(size_t k = 0; k < N_m; ++k){
                        const T x = h_m[0]*i;
                        const T y = h_m[1]*j;
                        const T z = h_m[2]*k;
                        f_m[index(i,j,k)] = x2y2z2(x,y,z);
                    }
                }
            }
        }
        
        void init_x2y2z2_deriv(){
            for(size_t i = 0; i < N_m; ++i){
                for(size_t j = 0; j < N_m; ++j){
                    for(size_t k = 0; k < N_m; ++k){
                        const T x = h_m[0]*i;
                        const T y = h_m[1]*j;
                        const T z = h_m[2]*k;
                        f_m[index(i,j,k)] = x2y2z2_deriv(x,y,z);
                    }
                }
            }
        }

        vector3d_t getMeshSpacing() const {return hInv_m;};

        T operator()(size_t i, size_t j, size_t k) const {return f_m[index(i,j,k)];};

        T& operator()(size_t i, size_t j, size_t k) {return f_m[index(i,j,k)];};

        T operator()(Index idx) const {return f_m[idx.ravel()];};

        T& operator()(Index idx) {return f_m[idx.ravel()];};

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
        const vector3d_t h_m;
        vector3d_t hInv_m;
        // Field data
        std::vector<T> f_m;
};


// Special overloaded operators for defined datastructures used in Field
template<typename T>
typename Field<T>::vector3d_t add_vector3d(typename Field<T>::vector3d_t &vec1,
                                            typename Field<T>::vector3d_t &vec2,
                                            typename Field<T>::vector3d_t &vec3) {
            return {vec1[0]+vec2[0]+vec3[0],
                    vec1[1]+vec2[1]+vec3[1],
                    vec1[2]+vec2[2]+vec3[2]};
}
    
template<typename T>
inline typename Field<T>::vector3d_t scale_vector3d(typename Field<T>::vector3d_t vec1, T scalar1){
    return {vec1[0]*scalar1, vec1[1]*scalar1, vec1[2]*scalar1};
}

#endif // field_h
