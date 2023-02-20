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

namespace Mesh {

enum Dim {X, Y, Z};

typedef Eigen::Vector3d vector3d_t;
typedef Eigen::Vector3i vector3i_t;
typedef Eigen::Matrix3d matrix3d_t;

// Taken from https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
        // unless the result is subnormal
        || std::fabs(x-y) < std::numeric_limits<T>::min();
}

template<typename T>
T relative_error(T x, T x_appr){ 
    if (almost_equal(x, x_appr, 2)){
        return 0.0;
    } else {
        return std::abs((x - x_appr) / x);
    }
}

struct Index {
    typedef std::tuple<size_t,size_t,size_t> idx_tuple_t;
    Index(idx_tuple_t idx_arg, vector3i_t N) : idx(idx_arg), n(N) {}
    Index(size_t raveled_idx) : idx(unravel(raveled_idx)) {}

    inline size_t ravel() const {return std::get<0>(idx)*n[1]*n[2] + std::get<1>(idx)*n[2] + std::get<2>(idx);}
    inline idx_tuple_t unravel(size_t raveled_idx) const {
        size_t i,j,k;
        i = raveled_idx / (n[1]*n[2]);
        j = raveled_idx / n[2];
        k = raveled_idx % n[2];
        return {i,j,k};
    }

    template<Dim D>
    inline Index get_shifted(size_t shift) const {
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
    vector3i_t n;
};

double gaussian(double x, double y, double z, double sigma = 1.0, double mu = 0.5) {
	double pi = std::acos(-1.0);
    double prefactor = (1 / std::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));
	double r2 = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);

	return -prefactor * std::exp(-r2 / (2 * sigma * sigma));
}

template<typename T>
T x2y2z2(T x, T y, T z) {
    return x*x*z+y*y+z*z*x;
}

template<typename T>
T xyz(T x, T y, T z) {
    return x*y*z;
}

template<typename T>
T x2y2z2_deriv(T x, T y, T z) {
    return 2.0*x+2.0*z;
    //return 2.0*z;
}



template <typename T>
class Field {
    public:
        // Constructs field on domain [0,1]^3
        Field(vector3d_t h, size_t nghost=1) : h_m(h), nghost_m(nghost) {
            hInv_m = h.cwiseInverse();
            N_m = vector3d_t::Ones().cwiseProduct(hInv_m).cast<int>();
            N_ext_m = N_m + vector3i_t::Constant(2*nghost);
            
            f_m = std::vector<T>(N_ext_m.prod());
        }

        void init(T val){
			// Create a random number generator
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<T> dist(0, 20);
			// Fill the vector with random numbers
			std::generate(f_m.begin(), f_m.end(), [&](){ return dist(gen); });
        }

		void init_with_function(std::function<T(T,T,T)> initFunc){
            for(size_t i = nghost_m; i < N_ext_m[0]-nghost_m; ++i){
                for(size_t j = nghost_m; j < N_ext_m[1]-nghost_m; ++j){
                    for(size_t k = nghost_m; k < N_ext_m[2]-nghost_m; ++k){
                        const T x = h_m[0]*i;
                        const T y = h_m[1]*j;
                        const T z = h_m[2]*k;
                        f_m[index(i,j,k)] = initFunc(x,y,z);
                    }
                }
            }
		}
		
        void init_x2y2z2(){
            init_with_function([](T x, T y, T z){ return x2y2z2(x,y,z); });
        }

        void init_xyz(){
            init_with_function([](T x, T y, T z){ return xyz(x,y,z); });
        }
        
        void init_x2y2z2_deriv(){
            init_with_function([](T x, T y, T z){ return x2y2z2_deriv(x,y,z); });
        }

        void init_gaussian(){
            init_with_function([](T x, T y, T z){ return gaussian(x,y,z); });
        }

        vector3d_t getInvMeshSpacing() const {return hInv_m;};
        vector3i_t getN() const {return N_m;};
        vector3i_t getN_ext() const {return N_ext_m;};

        T operator()(size_t i, size_t j, size_t k) const {return f_m[index(i,j,k)];};

        T& operator()(size_t i, size_t j, size_t k) {return f_m[index(i,j,k)];};

        T operator()(Index idx) const {return f_m[idx.ravel()];};

        T& operator()(Index idx) {return f_m[idx.ravel()];};

        void print(size_t slice_k) const {
            for(size_t i = nghost_m; i < N_ext_m[0]-nghost_m; ++i){
                for(size_t j = nghost_m; j < N_ext_m[1]-nghost_m; ++j){
                    std::cout << std::setprecision(2) << f_m[index(i,j,slice_k)] << " ";
                }
                std::cout << '\n';
            }
            std::cout << "\n\n" << std::endl;
        }


    private:
        inline size_t index(size_t i, size_t j, size_t k) const {
            return i*N_ext_m[1]*N_ext_m[2] + j*N_ext_m[2] + k;
        }

        // Number of grid points in each dimension
        vector3i_t N_m;
        size_t nghost_m;
        vector3i_t N_ext_m;
        // Mesh widths in each dimension
        const vector3d_t h_m;
        vector3d_t hInv_m;
        // Field data
        std::vector<T> f_m;
};

class MatrixField {
    public:
        MatrixField(vector3d_t h, size_t nghost=1) : h_m(h), nghost_m(nghost) {
            hInv_m = h.cwiseInverse();
            N_m = vector3d_t::Ones().cwiseProduct(hInv_m).cast<int>();
            N_ext_m = N_m + vector3i_t::Constant(2*nghost);
            f_m = std::vector<matrix3d_t>(N_ext_m.prod());
        }

        vector3d_t getInvMeshSpacing() const {return hInv_m;};
        vector3i_t getN() const {return N_m;};
        vector3i_t getN_ext() const {return N_ext_m;};

        matrix3d_t operator()(size_t i, size_t j, size_t k) const {return f_m[index(i,j,k)];};

        matrix3d_t& operator()(size_t i, size_t j, size_t k) {return f_m[index(i,j,k)];};

        matrix3d_t operator()(Index idx) const {return f_m[idx.ravel()];};

        matrix3d_t& operator()(Index idx) {return f_m[idx.ravel()];};

		// Initialize with hessian of gaussian defined by `gaussian`
		void initHess(bool gauss_init = false) {
            if(gauss_init){
			double mu = 0.5;
            for(size_t i = nghost_m; i < N_ext_m[0]-nghost_m; ++i){
                for(size_t j = nghost_m; j < N_ext_m[1]-nghost_m; ++j){
                        for(size_t k = nghost_m; k < N_ext_m[2]-nghost_m; ++k){
                        const double x = h_m[0]*i;
                        const double y = h_m[1]*j;
                        const double z = h_m[2]*k;
						f_m[index(i, j, k)].row(0) = vector3d_t(                                     
							((x - mu) * (x - mu) - 1.0) * gaussian(x, y, z), 
							(x - mu) * (y - mu) * gaussian(x, y, z),         
							(x - mu) * (z - mu) * gaussian(x, y, z));        
						f_m[index(i, j, k)].row(1) = vector3d_t(
							(x - mu) * (y - mu) * gaussian(x, y, z),         
							((y - mu) * (y - mu) - 1.0) * gaussian(x, y, z), 
							(y - mu) * (z - mu) * gaussian(x, y, z));        
						f_m[index(i, j, k)].row(2) = vector3d_t(
							(x - mu) * (z - mu) * gaussian(x, y, z),         
							(y - mu) * (z - mu) * gaussian(x, y, z),         
							((z - mu) * (z - mu) - 1.0) * gaussian(x, y, z));
                        }
                    }
                }
            } else {
            for(size_t i = nghost_m; i < N_ext_m[0]-nghost_m; ++i){
                for(size_t j = nghost_m; j < N_ext_m[1]-nghost_m; ++j){
                        for(size_t k = nghost_m; k < N_ext_m[2]-nghost_m; ++k){
                            const double x = h_m[0]*i;
                            const double y = h_m[1]*j;
                            const double z = h_m[2]*k;

                            f_m[index(i,j,k)] << 0.0, z, y,
                                                  z,0.0, x,
                                                  y, x, 0.0;
                        }
                    }
                }
            }
		}


        void print(size_t slice_k) const {
            for(size_t i = nghost_m; i < N_ext_m[0]-nghost_m; ++i){
                for(size_t j = nghost_m; j < N_ext_m[1]-nghost_m; ++j){
                    std::cout << std::setprecision(2) << f_m[index(i,j,slice_k)] << " ";
                }
                std::cout << '\n';
            }
            std::cout << "\n\n" << std::endl;
        }


    private:
        inline size_t index(size_t i, size_t j, size_t k) const {
            return i*N_ext_m[1]*N_ext_m[2] + j*N_ext_m[2] + k;
        }

        // Number of grid points in each dimension
        vector3i_t N_m;
        size_t nghost_m;
        vector3i_t N_ext_m;
        // Mesh widths in each dimension
        const vector3d_t h_m;
        vector3d_t hInv_m;
        // Field data
        std::vector<matrix3d_t> f_m;
};

} // Mesh

#endif // field_h
