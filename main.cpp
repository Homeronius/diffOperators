#include "hessian.h"

int main(){
    size_t N = 15;
    size_t nghost = 1;
    size_t N_ext = N+nghost;
    typename Field<double>::vector3d_t h{1e-1, 1e-1, 1e-1};
    Field<double> field(N+nghost, h);
    Field<double> field_result(N+nghost, h);
    Field<double> field_derivative(N+nghost, h);

    // Initialize
    field.init_x2y2z2();
    field_derivative.init_x2y2z2_deriv();

    // Construct differential operator
    DiffOpChain<Dim::X, double, DiffType::Forward, 
                DiffOpChain<Dim::Z, double, DiffType::Backward,
                            Field<double>>> xzDiff(field);

    //GeneralizedHessOp<double,DiffType::Centered,DiffType::Centered,DiffType::Centered> hessOp(field);

    double error = 0.0;
    for(size_t i = 2*nghost; i < N_ext-2*nghost; ++i){
        for(size_t j = 2*nghost; j < N_ext-2*nghost; ++j){
            for(size_t k = 2*nghost; k < N_ext-2*nghost; ++k){
                double approx_value = xzDiff(Index({i,j,k},N_ext));
                field_result(i,j,k) = approx_value;

                // Compute relative error
                error += relative_error(field_derivative(i,j,k), approx_value);
            }
        }
    }

    std::cout << "Relative Error: " << std::setw(10) << error << std::endl;

    return 0;
}
