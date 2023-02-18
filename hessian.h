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
// This only works if the container the operators are applied to has an overloaded `operator()`
template<Dim D, typename T, DiffType Diff, class C>
class DiffOpChain {
    public: 
        DiffOpChain(Field<T>& field) : f_m(field), hInvVector_m(field.getMeshSpacing()), leftOp(field) {}
        
        inline T operator()(Index idx) const {
            if constexpr (Diff == DiffType::Centered) {
                return centered_stencil<D,T,C>(idx, hInvVector_m[D], leftOp);
            } else if constexpr (Diff == DiffType::Forward) {
                return forward_stencil<D,T,C>(idx, hInvVector_m[D], leftOp);
            } else if constexpr (Diff == DiffType::Backward) {
                return backward_stencil<D,T,C>(idx, hInvVector_m[D], leftOp);
            }
        }

    private:
        Field<T>& f_m;
        typename Field<T>::vector3d_t hInvVector_m;
        C leftOp;
};

//template<typename T, DiffType DiffX, DiffType DiffY, DiffType DiffZ>
//class GeneralizedHessOp {
    //public:
        //GeneralizedHessOp(Field<T> field) : f_m(field){
            //// Define operators contained in Hessian
            //// Row 1
            //DiffOpChain<Dim::X,T,DiffX,
            //DiffOpChain<Dim::X,T,DiffX,Field<T>>> hess_xx(f_m);

            //DiffOpChain<Dim::X,T,DiffX,
            //DiffOpChain<Dim::Y,T,DiffY,Field<T>>> hess_xy(f_m);

            //DiffOpChain<Dim::X,T,DiffX,
            //DiffOpChain<Dim::Z,T,DiffZ,Field<T>>> hess_xz(f_m);
            //gen_row1_m = [hess_xx, hess_xy, hess_xz, this](Index idx){
                //return add_vector3d<T>(scale_vector3d(xvector_m,hess_xx(idx)),
                                        //scale_vector3d(yvector_m,hess_xy(idx)),
                                        //scale_vector3d(zvector_m,hess_xz(idx)));
            //};

            //// Row 2
            //DiffOpChain<Dim::Y,T,DiffY,
            //DiffOpChain<Dim::X,T,DiffX,Field<T>>> hess_yx(f_m);

            //DiffOpChain<Dim::Y,T,DiffY,
            //DiffOpChain<Dim::Y,T,DiffY,Field<T>>> hess_yy(f_m);

            //DiffOpChain<Dim::Y,T,DiffY,
            //DiffOpChain<Dim::Z,T,DiffZ,Field<T>>> hess_yz(f_m);
            //gen_row2_m = [hess_yx, hess_yy, hess_yz, this](Index idx){
                //return add_vector3d<T>(scale_vector3d(xvector_m,hess_yx(idx)),
                                        //scale_vector3d(yvector_m,hess_yy(idx)),
                                        //scale_vector3d(zvector_m,hess_yz(idx)));
            //};
            
            //// Row 3
            //DiffOpChain<Dim::Z,T,DiffZ,
            //DiffOpChain<Dim::X,T,DiffX,Field<T>>> hess_zx(f_m);

            //DiffOpChain<Dim::Z,T,DiffZ,
            //DiffOpChain<Dim::Y,T,DiffY,Field<T>>> hess_zy(f_m);

            //DiffOpChain<Dim::Z,T,DiffZ,
            //DiffOpChain<Dim::Z,T,DiffZ,Field<T>>> hess_zz(f_m);
            //gen_row3_m = [hess_zx, hess_zy, hess_zz, this](Index idx){
                //return add_vector3d<T>(scale_vector3d(xvector_m,hess_zx(idx)),
                                        //scale_vector3d(yvector_m,hess_zy(idx)),
                                        //scale_vector3d(zvector_m,hess_zz(idx)));
            //};
        //}
        
        //inline typename Field<T>::matrix3d_t operator()(Index idx) const {
            //typename Field<T>::vector3d_t row_1, row_2, row_3;
            //row_1 = gen_row1_m(idx);
            //row_2 = gen_row2_m(idx);
            //row_3 = gen_row3_m(idx);
            //return {row_1, row_2, row_3};
        //}

    //private:
        //Field<T>& f_m;
        //typename Field<T>::vector3d_t hInvVector_m;
        //std::function<T(Index)> gen_row1_m, gen_row2_m, gen_row3_m;
        //const typename Field<T>::vector3d_t xvector_m = {1.0, 0.0, 0.0};
        //const typename Field<T>::vector3d_t yvector_m = {0.0, 1.0, 0.0};
        //const typename Field<T>::vector3d_t zvector_m = {0.0, 0.0, 1.0};
//};
























#endif // hessian_h
