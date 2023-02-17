#ifndef hessian_h
#define hessian_h

#include <functional>
#include "field.h"


enum DiffType {Centered, Forward, Backward};

// Stencil definitions along a template specified dimension
template<Dim D, typename T, class Callable>
inline T centered_stencil(Index idx, T hInv, Callable F){
    return 0.5 * hInv * (-1.0*F(idx.get_shifted<D>(-1)) + F(idx.get_shifted<D>(1)));
}

template<Dim D, typename T, class Callable>
inline T forward_stencil(Index idx, T hInv, Callable F){
    return hInv * (-1.5*F(idx) + 2.0*F(idx.get_shifted<D>(1)) - 0.5*F(idx.get_shifted<D>(2)));
}

template<Dim D, typename T, class Callable>
inline T backward_stencil(Index idx, T hInv, Callable F){
    return hInv * (1.5*F(idx) - 2.0*F(idx.get_shifted<D>(-1)) + 0.5*F(idx.get_shifted<D>(-2)));
}

// Specialization to chain stencil operators
template<Dim D, typename T, DiffType Diff, class C>
class DiffOpChain {
    public: 
        DiffOpChain(Field<T>& field)  : f_m(field), hInv_m(field.get_hInv()), leftOp(field) {}
        
        inline T operator()(Index idx) const {
            if constexpr (Diff == DiffType::Centered) {
                return centered_stencil<D,T,C>(idx, hInv_m[D], leftOp);
            } else if constexpr (Diff == DiffType::Forward) {
                return forward_stencil<D,T,C>(idx, hInv_m[D], leftOp);
            } else if constexpr (Diff == DiffType::Backward) {
                return backward_stencil<D,T,C>(idx, hInv_m[D], leftOp);
            }
        }

    private:
        Field<T>& f_m;
        std::array<T, 3> hInv_m;
        C leftOp;
};

#endif // hessian_h
