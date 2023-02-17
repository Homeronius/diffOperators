#ifndef hessian_h
#define hessian_h

#include "field.h"

double gaussian(double x, double y, double z, double sigma = 1.0, double mu = 0.5) { 
    double pi = std::acos(-1.0);                                                     
    double prefactor =                                                               
        (1 / std::sqrt(2 * 2 * 2 * pi * pi * pi)) * (1 / (sigma * sigma * sigma));   
    double r2 = (x - mu) * (x - mu) + (y - mu) * (y - mu) + (z - mu) * (z - mu);     
                                                                                     
    return -prefactor * std::exp(-r2 / (2 * sigma * sigma));                         
}                                                                                    


enum DiffType {Forward, Backward, Centered};
enum Dim {X, Y, Z};


// Specializations for the different differntial stencils
template<Dim D, typename T, DiffType Diff>
class DiffOpBase {
    public: 
        DiffOpBase(Field<T>& field)  : f_m(field), hInv_m(field.get_hInv()) {}
        
        inline T shift_f(size_t shift, size_t i, size_t j, size_t k) const {
            if constexpr (D == Dim::X) {
                return f_m(i+shift,j,k);
            } else if constexpr (D == Dim::Y) {
                return f_m(i,j+shift,k);
            } else if constexpr (D == Dim::Z) {
                return f_m(i,j,k+shift);
            }
        }

       
        inline T operator()(size_t i, size_t j, size_t k) const {
            if constexpr (Diff == DiffType::Centered) {
                return 0.5 * hInv_m[D] * (-1.0*shift_f(-1,i,j,k) + shift_f(1,i,j,k));
            } else if constexpr (Diff == DiffType::Forward) {
                return 0.5 * hInv_m[D] * (-1.5*shift_f(0,i,j,k) + 2.0*shift_f(1,i,j,k) -
                                          0.5*shift_f(2,i,j,k));
            } else if constexpr (Diff == DiffType::Backward) {
                return 0.5 * hInv_m[D] * (1.5*shift_f(0,i,j,k) - 2.0*shift_f(-1,i,j,k) +
                                          0.5*shift_f(-2,i,j,k));
            }
        }

    private:
        Field<T>& f_m;
        std::array<T, 3> hInv_m;
};

//template<Dim D, typename T, class C>
//class DiffOp {

//};

#endif // hessian_h
